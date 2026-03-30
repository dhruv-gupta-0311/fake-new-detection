import pandas as pd

import nltk
import os
venv_path = os.path.join(os.getcwd(), '.venv', 'nltk_data')
nltk.data.path.append(venv_path)
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
process = DataProcessor()

# print("Processing 72k rows from dataset")
# df_subset = process.prep_dataframe('WELFake_Dataset.csv', nrows=72000)
# df_subset.to_csv('final_processed.csv', index=False)
# print("final Sample of processed data saved to 'final_processed.csv'.")
trainer = ModelTrainer()
X_train, X_test, Y_train, Y_test = trainer.prepare_data('final_processed.csv')
trainer.train_model_logistic(X_train, Y_train)
accuracy = trainer.evaluation_model(X_test, Y_test, X_train, Y_train)
trainer.plot_pr_curve(X_test, Y_test)
trainer.save_model()
