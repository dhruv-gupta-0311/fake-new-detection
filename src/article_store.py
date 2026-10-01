import hashlib
import os
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer


class ArticleStore:
    def __init__(self, persist_dir: str = "data/chroma_db", collection_name: str = "news_articles"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        existing_collections = [collection.name for collection in self.client.list_collections()]
        if collection_name not in existing_collections:
            self.collection = self.client.create_collection(name=collection_name)
        else:
            self.collection = self.client.get_collection(name=collection_name)

    def _article_id(self, article: Dict[str, Any]) -> str:
        url = str(article.get("url") or article.get("title") or "")
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def normalize_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        source = article.get("source", "")
        if isinstance(source, dict):
            source_name = source.get("name", "")
        else:
            source_name = str(source or "")

        title = str(article.get("title") or "").strip()
        content = str(article.get("content") or article.get("description") or "").strip()

        if not title and not content:
            return {}

        return {
            "id": self._article_id(article),
            "title": title,
            "content": content,
            "source": source_name,
            "url": str(article.get("url") or ""),
            "published_at": str(article.get("publishedAt") or article.get("published_at") or ""),
            "fetched_at": str(article.get("fetched_at") or ""),
            "topic": str(article.get("topic") or ""),
        }

    def build_document(self, normalized_article: Dict[str, Any]) -> str:
        title = normalized_article.get("title", "")
        content = normalized_article.get("content", "")
        if title and content:
            return f"{title}\n\n{content}"
        return title or content

    def build_metadata(self, normalized_article: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": normalized_article.get("title", ""),
            "source": normalized_article.get("source", ""),
            "url": normalized_article.get("url", ""),
            "published_at": normalized_article.get("published_at", ""),
            "fetched_at": normalized_article.get("fetched_at", ""),
            "topic": normalized_article.get("topic", ""),
        }

    def add_articles(self, articles: List[Dict[str, Any]]) -> int:
        valid_articles = []
        for article in articles:
            normalized = self.normalize_article(article)
            if not normalized:
                continue
            valid_articles.append(normalized)

        if not valid_articles:
            return 0

        ids = [article["id"] for article in valid_articles]
        documents = [self.build_document(article) for article in valid_articles]
        metadatas = [self.build_metadata(article) for article in valid_articles]
        embeddings = self.encoder.encode(documents).tolist()

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def query_articles(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        # Guard: ChromaDB raises if collection is empty or n_results > count
        count = self.collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)

        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = []
        for idx, document in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            output.append({
                "title": metadata.get("title", ""),
                "content": document,
                "source": metadata.get("source", ""),
                "url": metadata.get("url", ""),
                "score": float(distances[idx]) if idx < len(distances) else None,
            })
        return output

    def count_articles(self) -> int:
        return self.collection.count()
