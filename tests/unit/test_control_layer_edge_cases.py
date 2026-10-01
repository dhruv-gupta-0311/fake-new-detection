"""Edge case tests for EvidencePipeline."""
import pytest

from src.article_store import ArticleStore
from src.control_layer import EvidencePipeline


class DummyFetcher:
    def __init__(self, articles):
        self.articles = articles

    def fetch_by_query(self, query, days_back=7, max_articles=20):
        return self.articles


def test_ingest_empty_fetch_result(tmp_path):
    """When fetcher returns [], ingest_query reports 0 fetched/inserted."""
    store = ArticleStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_articles"
    )
    pipeline = EvidencePipeline(fetcher=DummyFetcher([]), store=store)

    result = pipeline.ingest_query("empty topic")

    assert result["fetched"] == 0
    assert result["inserted"] == 0
    assert result["skipped"] == 0
    assert store.count_articles() == 0


def test_retrieve_before_ingest_returns_empty(tmp_path):
    """Querying evidence before any ingest returns an empty list, not an error."""
    store = ArticleStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_articles"
    )
    pipeline = EvidencePipeline(fetcher=DummyFetcher([]), store=store)

    evidence = pipeline.retrieve_evidence("some claim", n_results=5)

    assert evidence == []


def test_ingest_articles_with_bad_data_counted_as_skipped(tmp_path):
    """Articles that fail normalization are reflected in the skipped count."""
    bad_articles = [
        {"title": "", "content": ""},   # both empty → skipped
        {"source": "Reuters"},          # no title/content → skipped
        {                               # valid
            "title": "Real article",
            "content": "Actual content",
            "url": "https://example.com/real",
        },
    ]
    store = ArticleStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_articles"
    )
    pipeline = EvidencePipeline(fetcher=DummyFetcher(bad_articles), store=store)

    result = pipeline.ingest_query("mixed quality articles")

    assert result["fetched"] == 3
    assert result["inserted"] == 1
    assert result["skipped"] == 2


def test_retrieve_evidence_returns_correct_article(tmp_path):
    """After ingesting, retrieve_evidence returns the relevant article."""
    articles = [
        {
            "title": "Federal Reserve raises rates",
            "content": "The Fed raised interest rates by 25 basis points.",
            "source": "Reuters",
            "url": "https://example.com/fed",
        },
        {
            "title": "SpaceX launches new rocket",
            "content": "SpaceX successfully launched a Starship prototype.",
            "source": "TechCrunch",
            "url": "https://example.com/spacex",
        },
    ]
    store = ArticleStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_articles"
    )
    pipeline = EvidencePipeline(fetcher=DummyFetcher(articles), store=store)
    pipeline.ingest_query("test", max_articles=2)

    evidence = pipeline.retrieve_evidence("Federal Reserve interest rates", n_results=1)

    assert len(evidence) == 1
    assert "Federal Reserve" in evidence[0]["title"]


def test_ingest_query_string_preserved(tmp_path):
    """The query string is echoed back verbatim in the result dict."""
    store = ArticleStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_articles"
    )
    pipeline = EvidencePipeline(fetcher=DummyFetcher([]), store=store)

    query = "  Specific Query With Spaces  "
    result = pipeline.ingest_query(query)

    assert result["query"] == query
