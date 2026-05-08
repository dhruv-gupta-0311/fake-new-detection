# scripts/push_model.py
from huggingface_hub import HfApi, login
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

def push_bert_model():
    login(token=os.getenv("HF_Token"))
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    api.create_repo(
        repo_id="Dhruv-0113/fakenews-distilbert-welfake",
        private=True,
        exist_ok=True
    )
    
    print("Loading model from local...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'models/distilbert_finetuned'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'models/distilbert_finetuned'
    )
    
    print("Pushing to HuggingFace Hub...")
    model.push_to_hub("Dhruv-0113/fakenews-distilbert-welfake")
    tokenizer.push_to_hub("Dhruv-0113/fakenews-distilbert-welfake")
    
    print("Done. Model available at:")
    print("https://huggingface.co/Dhruv-0113/fakenews-distilbert-welfake")

if __name__ == "__main__":
    push_bert_model()