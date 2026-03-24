import nltk
import os
def download_nltk_requirements():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nltk_data_path = os.path.join(base_dir, '.venv', 'nltk_data')
    
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
        
    print(f"Downloading assets to: {nltk_data_path}")
    
    # 2. Force the download to that specific directory
    resources = ['stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        nltk.download(res, download_dir=nltk_data_path)
    
    print("\nSetup Complete. Check .venv/nltk_data/corpora now.")

if __name__ == "__main__":
    download_nltk_requirements()