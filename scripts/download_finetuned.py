from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
load_dotenv()
def download_bert_model():
    model_path = "models/distilbert_finetuned"
    if os.path.exists(model_path):
        print("Model already exists. Skipping download.")
        return
    print("Downloading DistilBERT fine-tuned model...")
    snapshot_download(
        repo_id="Dhruv-0113/fakenews-distilbert-welfake",
        local_dir=model_path,
        token=os.getenv("HF_Token")
    )
    print(f"Model downloaded to {model_path}")
    
