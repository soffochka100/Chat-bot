
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from typing import Optional, Tuple

bert_model = None
bert_tokenizer = None
bert_label_map = None


def load_bert_model(model_path: str = "bert_intent_model"):
    """
    Загрузка дообученной BERT модели
    """
    global bert_model, bert_tokenizer, bert_label_map
    
    try:
        if os.path.exists(model_path):
            print(f"Загрузка BERT из {model_path}...")
            bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
            bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            bert_model.eval()
            
            if os.path.exists(f"{model_path}/label_map.json"):
                with open(f"{model_path}/label_map.json", 'r', encoding='utf-8') as f:
                    bert_label_map = json.load(f)
                    bert_label_map = {int(k): v for k, v in bert_label_map.items()}
            else:
                bert_label_map = {0: "greeting", 1: "weather", 2: "goodbye", 
                                 3: "addition", 4: "set_name", 5: "unknown"}
            
            print(f" BERT модель успешно загружена из {model_path}")
            return True
        else:
            print(f" BERT модель не найдена в {model_path}")
            print("   Запустите python train_bert.py для обучения модели")
            return False
    except Exception as e:
        print(f" Ошибка загрузки BERT модели: {e}")
        return False


def predict_intent_bert(text: str) -> Tuple[Optional[str], float]:
    """
    Предсказание интента с помощью BERT
    Возвращает (интент, уверенность)
    """
    global bert_model, bert_tokenizer, bert_label_map
    
    if bert_model is None or bert_tokenizer is None:
        return None, 0.0
    
    try:
        inputs = bert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        intent = bert_label_map.get(predicted_class, "unknown")
        
        return intent, confidence
        
    except Exception as e:
        print(f" Ошибка при предсказании BERT: {e}")
        return None, 0.0


def is_bert_available() -> bool:
    """Проверка, загружена ли BERT модель"""
    return bert_model is not None