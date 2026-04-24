import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
extra_stop_words = ['reuters', 'video', 'image', 'via', '2016', 'october', 'november', 'donald', 'hillary']
stop_words_final = list(ENGLISH_STOP_WORDS.union(extra_stop_words))
class ModelTrainer:
    def __init__(self, max_features=50000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words=stop_words_final)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def prepare_data(self, file_path):
        print(f"loading cleaned data from {file_path}")
        df = pd.read_csv(file_path)
        df.dropna(subset=['content'], inplace=True)
        X = df['content']
        Y = df['label']
        X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
        print("vectorizing using TF-IDF")
        X_train = self.vectorizer.fit_transform(X_train_raw)
        X_test = self.vectorizer.transform(X_test_raw)
        return X_train, X_test, Y_train, Y_test
    
    def train_model_logistic(self, X_train, Y_train):
        print("training logistic regression model")
        self.model.fit(X_train, Y_train)
        print("model training complete")
        
    def plot_pr_curve(self, X_test, Y_test):
        Y_scores = self.model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(Y_test, Y_scores)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve(AUC={pr_auc:.2f})')
        plt.xlabel('Recall (Ability to catch Fake News)')
        plt.ylabel('Precision (Ability to be correct about Fake News)')
        plt.title('Precision-Recall Curve: Fake News Detection')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
        print(f"Precision-Recall AUC: {pr_auc:.2f}")
        
    
    def evaluation_model(self, X_test, Y_test, X_train, Y_train):
        print("evaluating model performance")
        Y_pred = self.model.predict(X_test)
        Y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(Y_train, Y_train_pred)
        test_accuracy = accuracy_score(Y_test, Y_pred)
        report = classification_report(Y_test, Y_pred)
        matrix = confusion_matrix(Y_test, Y_pred)
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print("classification report:\n", report)
        print("confusion matrix:\n", matrix)
    
    def save_model(self): 
        joblib.dump(self.model, 'models/logistic_model.joblib')
        joblib.dump(self.vectorizer, 'models/tfidf_vectorized.joblib')
        print("model, vectorized saved to models directory")
        
