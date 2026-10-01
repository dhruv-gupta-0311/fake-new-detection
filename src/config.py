import os
from dotenv import load_dotenv
from dataclasses import dataclass
load_dotenv()
@dataclass
class Config:
    newsapi_key: str = os.getenv('NEWS_API_SECRET', '')
    gemini_key: str = os.getenv('GEMINI_API_KEY', '')
    hf_key: str = os.getenv('HF_Token', '')
    newsapi_daily_limit: int = 900
    max_articles_per_query: int = 100
    bert_model_path: str = "models/distilbert_finetuned"
    lr_model_path: str = "models/logistic_model.joblib"
    vectorizer_path: str = "models/tfidf_vectorized.joblib"
    tfidf_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.65
    nli_confidence_threshold: float = 0.75
    label_confidence_threshold: float = 0.80
    retrain_article_threshold: int = 5000
    raw_data_path: str = "data/WELFake_Dataset.csv"
    processed_data_path: str = "data/final_processed.csv"
    live_data_path: str = "data/live_training_data.csv"
    chroma_persist_dir: str = "data/chroma_db"
    gemini_cache_dir: str = "data/cache/gemini"
    news_cache_dir: str = "data/cache/newsapi"
    api_usage_file: str = "data/api_usage.json"
    api_host: str = "localhost"
    api_port: int = 8000
    fetch_schedule_time: str = "02:00"
    fetch_topics: tuple = (
        "politics government policy",
        "economy federal reserve inflation",
        "science research climate",
        "international relations diplomacy",
        "health medicine FDA",
        "technology artificial intelligence",
        "election voting democracy",
    )
config = Config()
