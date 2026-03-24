import nltk
import os
venv_path = os.path.join(os.getcwd(), '.venv', 'nltk_data')
nltk.data.path.append(venv_path)
from src.data_processor import DataProcessor
process = DataProcessor()
# print("Data process test1")
# #first manual data test
# dirty_sample = "BREAKING: Check this out at https://fake-news.com/scam !!! It's UNBELIEVABLE."
# cleaned_sample = process.clean_text(dirty_sample)
# print(f"Original: {dirty_sample}")
# print(f"Cleaned: {cleaned_sample}")
print("Processing 10k rows from dataset")
df_subset = process.prep_dataframe('WELFake_Dataset.csv', nrows=10000)
df_subset.to_csv('processed_sample.csv', index=False)
print("Sample of processed data saved to 'processed_sample.csv'.")