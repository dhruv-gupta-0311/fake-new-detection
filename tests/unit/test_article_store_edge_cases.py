"""Edge case tests for ArticleStore."""
import pytest

from src.article_store import ArticleStore


def test_add_empty_article_list(tmp_path):
    """Test adding an empty list of articles."""
    persist_dir = tmp_path / "chroma_empty"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    inserted = store.add_articles([])
    assert inserted == 0
    assert store.count_articles() == 0


def test_add_articles_with_missing_fields(tmp_path):
    """Test articles with missing or empty fields are filtered out."""
    persist_dir = tmp_path / "chroma_missing"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    articles = [
        {"title": "", "content": ""},  # both empty - should be filtered
        {"title": "Valid title", "content": ""},  # has title - should be kept
        {"title": "", "content": "Valid content"},  # has content - should be kept
        {"source": "Reuters"},  # no title or content - should be filtered
    ]

    inserted = store.add_articles(articles)
    assert inserted == 2  # only articles with title OR content
    assert store.count_articles() == 2


def test_add_duplicate_articles_upsert(tmp_path):
    """Test that duplicate articles (same URL) are updated, not added."""
    persist_dir = tmp_path / "chroma_duplicates"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    article_v1 = {
        "title": "Original title",
        "content": "Original content",
        "source": "Reuters",
        "url": "https://example.com/article1",
        "publishedAt": "2026-07-01T10:00:00Z",
    }

    # Add first version
    inserted1 = store.add_articles([article_v1])
    assert inserted1 == 1
    assert store.count_articles() == 1

    # Add "updated" version with same URL
    article_v2 = {**article_v1, "title": "Updated title", "content": "Updated content"}
    inserted2 = store.add_articles([article_v2])
    assert inserted2 == 1
    assert store.count_articles() == 1  # still 1, upserted not added

    # Query should return updated version
    results = store.query_articles("Updated title", n_results=1)
    assert len(results) == 1
    assert "Updated title" in results[0]["title"]


def test_query_empty_collection(tmp_path):
    """Test querying an empty collection."""
    persist_dir = tmp_path / "chroma_empty_query"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    results = store.query_articles("anything", n_results=5)
    assert results == []


def test_query_with_empty_string(tmp_path):
    """Test querying with empty or whitespace-only string."""
    persist_dir = tmp_path / "chroma_whitespace"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    articles = [
        {
            "title": "Test article",
            "content": "Some content here",
            "url": "https://example.com/test",
        }
    ]
    store.add_articles(articles)

    # Empty string query
    results = store.query_articles("", n_results=5)
    assert results == []

    # Whitespace-only query
    results = store.query_articles("   \t\n  ", n_results=5)
    assert results == []


def test_query_no_matching_results(tmp_path):
    """Test query that doesn't match any articles."""
    persist_dir = tmp_path / "chroma_no_match"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    articles = [
        {
            "title": "Federal Reserve interest rates",
            "content": "Central bank announcement about rates",
            "source": "Reuters",
            "url": "https://example.com/fed",
        }
    ]
    store.add_articles(articles)

    # Query for completely unrelated topic
    results = store.query_articles("quantum computing breakthrough", n_results=5)
    # ChromaDB will still return results but with low relevance scores
    # We should validate that results are returned but may have high distances
    assert isinstance(results, list)


def test_query_with_n_results_larger_than_collection(tmp_path):
    """Test requesting more results than articles in collection."""
    persist_dir = tmp_path / "chroma_small"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    articles = [
        {
            "title": "Article 1",
            "content": "Content about Federal Reserve",
            "url": "https://example.com/1",
        },
        {
            "title": "Article 2",
            "content": "Content about interest rates",
            "url": "https://example.com/2",
        },
    ]
    store.add_articles(articles)

    # Request 100 results but only 2 in collection
    results = store.query_articles("Federal Reserve", n_results=100)
    assert len(results) <= 2  # ChromaDB returns only what exists


def test_add_articles_with_various_source_formats(tmp_path):
    """Test normalizing articles with source as dict vs string."""
    persist_dir = tmp_path / "chroma_sources"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    articles = [
        {
            "title": "Article with dict source",
            "content": "Content here",
            "source": {"name": "Reuters", "id": "reuters"},
            "url": "https://example.com/1",
        },
        {
            "title": "Article with string source",
            "content": "Content here",
            "source": "BBC News",
            "url": "https://example.com/2",
        },
        {
            "title": "Article with no source",
            "content": "Content here",
            "url": "https://example.com/3",
        },
    ]

    inserted = store.add_articles(articles)
    assert inserted == 3

    results = store.query_articles("Article", n_results=10)
    sources = [r["source"] for r in results]
    assert "Reuters" in sources
    assert "BBC News" in sources
    assert "" in sources  # empty source for third article


def test_count_articles(tmp_path):
    """Test counting articles in collection."""
    persist_dir = tmp_path / "chroma_count"
    store = ArticleStore(persist_dir=str(persist_dir), collection_name="test_articles")

    assert store.count_articles() == 0

    store.add_articles([
        {"title": "A", "content": "content", "url": "https://a.com"},
        {"title": "B", "content": "content", "url": "https://b.com"},
        {"title": "C", "content": "content", "url": "https://c.com"},
    ])

    assert store.count_articles() == 3
