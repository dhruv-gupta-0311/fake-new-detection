# src/control_layer.py
from src.article_fetcher import NewsFetcher
from src.article_store import ArticleStore


class EvidencePipeline:
    def __init__(self, fetcher: NewsFetcher | None = None, store: ArticleStore | None = None):
        self.fetcher = fetcher or NewsFetcher()
        self.store = store or ArticleStore()

    def ingest_query(
        self,
        query: str,
        days_back: int = 7,
        max_articles: int = 20,
    ) -> dict:
        raw_articles = self.fetcher.fetch_by_query(
            query,
            days_back=days_back,
            max_articles=max_articles,
        )

        inserted = self.store.add_articles(raw_articles)

        return {
            "query": query,
            "fetched": len(raw_articles),
            "inserted": inserted,
            "skipped": len(raw_articles) - inserted,
        }

    def retrieve_evidence(self, claim: str, n_results: int = 5) -> list[dict]:
        return self.store.query_articles(claim, n_results=n_results)