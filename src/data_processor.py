import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
class DataProcessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        try:
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            print("Downloading NLTK WordNet data...")
            nltk.download('wordnet')
            self.lemmatizer = WordNetLemmatizer()
            
    def prep_dataframe(self, file_path, nrows=1000):
        #loading dataset and selecting first 1000 rows to merge title and text columns
        df = pd.read_csv(file_path, low_memory=False, nrows=nrows)
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        df['content'] = df['title'] + ' ' + df['text']#merge title, text safely(handling NaN values)
        df = df[['content', 'label']]#selecting content, label columns
        print(f"Dataframe fed to preprocessing pipeline to {len(df)} rows")
        df['content'] = df['content'].apply(self.clean_text)
        return df
    def clean_text(self, text):
        text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        tokens = text.split()
        cleaned_tokens = [self.lemmatizer.lemmatize(word)
                          for word in tokens
                          if word not in self.stop_words]#removing stopwords and lemmematizing
        return " ".join(cleaned_tokens)
    