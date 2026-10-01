from pathlib import Path

from src.article_store import ArticleStore
from src.control_layer import EvidencePipeline


class DummyFetcher:
    def __init__(self, articles):
        self.articles = articles

    def fetch_by_query(self, query, days_back=7, max_articles=20):
        return self.articles


def test_evidence_pipeline_ingest_and_retrieve(tmp_path):
    persist_dir = tmp_path / "chroma_control_test"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_news_articles")

    raw_articles = [
        {
            "title": "Federal Reserve keeps interest rates steady",
            "content": "The Federal Reserve announced it will keep interest rates unchanged.",
            "source": "Reuters",
            "url": "https://example.com/fed-rates",
            "publishedAt": "2026-07-01T10:00:00Z",
            "fetched_at": "2026-07-02T12:00:00Z",
            "topic": "economy",
        }
    ]

    fetcher = DummyFetcher(raw_articles)
    pipeline = EvidencePipeline(fetcher=fetcher, store=store)

    result = pipeline.ingest_query("Federal Reserve interest rates", max_articles=1)

    assert result["query"] == "Federal Reserve interest rates"
    assert result["fetched"] == 1
    assert result["inserted"] == 1
    assert result["skipped"] == 0

    evidence = pipeline.retrieve_evidence("Federal Reserve interest rates", n_results=1)
    assert len(evidence) == 1
    assert evidence[0]["source"] == "Reuters"
    assert "Federal Reserve" in evidence[0]["title"]
