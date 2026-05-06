from transformers import pipeline
import torch
class BertPredictor:
    def __init__(self, model_path = 'models/distilbert_finetuned'):
        self.classifier = pipeline("text-classification"
                                   ,model=model_path,
                                   tokenizer=model_path,
                                   truncation=True,
                                   max_length=512,
                                   device=0 if torch.cuda.is_available() else -1)
        self.label_map = {0: 'Real', 1: 'Fake'}
    def predict(self, text):
            result = self.classifier(text[:2000])[0]
            label_str = result['label']
            score = result['score']
            if label_str == 'Real':
                prediction = 0
                probability = [score, 1 - score]
            else:
                prediction = 1
                probability = [1 - score, score]
            return prediction, probability, score
        