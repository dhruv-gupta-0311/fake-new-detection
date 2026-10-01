from pathlib import Path

from src.article_store import ArticleStore


def test_add_and_query_articles(tmp_path):
    persist_dir = tmp_path / "chroma_test"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_news_articles")

    articles = [
        {
            "title": "Federal Reserve keeps interest rates steady",
            "content": "The Federal Reserve announced it will keep interest rates unchanged as inflation cools.",
            "source": "Reuters",
            "url": "https://example.com/fed-rates",
            "publishedAt": "2026-07-01T10:00:00Z",
            "fetched_at": "2026-07-02T12:00:00Z",
            "topic": "economy",
        }
    ]

    inserted = store.add_articles(articles)
    assert inserted == 1

    results = store.query_articles("Federal Reserve interest rates", n_results=1)
    assert len(results) == 1
    assert results[0]["source"] == "Reuters"
    assert "Federal Reserve" in results[0]["title"]
