import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json

MODEL_NAME = "cointegrated/rubert-tiny2"
OUTPUT_DIR = "bert_intent_model"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert():
    print("=" * 60)
    print("Дообучение BERT для классификации интентов")
    print("=" * 60)
    
    if not os.path.exists("dataset.csv"):
        print(" ОШИБКА: Файл dataset.csv не найден!")
        print("   Создайте файл dataset.csv с примерами интентов.")
        return None
    
    df = pd.read_csv("dataset.csv", encoding='utf-8')
    print(f" Загружен датасет из dataset.csv: {len(df)} примеров")
    print(f"\n Распределение по интентам:")
    intent_counts = df['intent'].value_counts()
    for intent, count in intent_counts.items():
        print(f"   {intent}: {count} примеров")
    
    unique_intents = sorted(df['intent'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_intents)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"\n Всего интентов: {len(unique_intents)}")
    print(f"   {label2id}")
    
    df['label'] = df['intent'].map(label2id)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label'].tolist()
    )
    
    print(f"\n Обучающая выборка: {len(train_texts)} примеров")
    print(f" Валидационная выборка: {len(val_texts)} примеров")
    
    print(f"\n Загрузка модели {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_intents),
        id2label=id2label,
        label2id=label2id
    )
    
    print(" Токенизация данных...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=35,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    print("\n НАЧАЛО ОБУЧЕНИЯ BERT...")
    print("   (Это может занять 5-10 минут)")
    trainer.train()
    

    print(f"\n Сохранение модели в {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    with open(f"{OUTPUT_DIR}/label_map.json", 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    
    print("\n Финальная оценка модели:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print(" ТЕСТИРОВАНИЕ BERT МОДЕЛИ:")
    print("=" * 60)
    
    test_examples = [
        "привет",
        "какая погода в москве",
        "сложи 5 и 3",
        "пока",
        "меня зовут Антон",
        "что ты умеешь",
        "сколько времени",
        "какое сегодня число",
        "спасибо",
        "повтори",
        "отмена"
    ]
    
    model.eval()
    results = []
    for example in test_examples:
        inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0][pred_class].item()
        
        intent = id2label[pred_class]
        results.append((example, intent, confidence))
        print(f"\n'{example}' → {intent} (уверенность: {confidence:.2%})")
    
    print("\n" + "=" * 60)
    print("СТАТИСТИКА:")
    high_conf = sum(1 for _, _, conf in results if conf > 0.7)
    print(f"  Примеров с уверенностью >70%: {high_conf}/{len(results)}")
    
    print(f"\n Модель сохранена в {OUTPUT_DIR}")
    return model

if __name__ == "__main__":
    train_bert()