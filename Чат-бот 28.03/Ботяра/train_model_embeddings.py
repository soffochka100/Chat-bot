import pandas as pd
import numpy as np
import spacy
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

try:
    nlp = spacy.load("ru_core_news_sm")
except:
    print("Установите spaCy модель: python -m spacy download ru_core_news_sm")
    exit(1)

def preprocess_text(text):
    """
    Предобработка текста: лемматизация, удаление стоп-слов и пунктуации
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.lemma_)
    return " ".join(tokens)

def get_word_embedding(text):
    """
    Получение векторного представления текста через Word Embeddings (spaCy)
    Усреднение векторов слов
    """
    doc = nlp(text.lower())
    if len(doc) == 0:
        return np.zeros(96)
    return doc.vector

def create_dataset():
    """
    Создание датасета с четким разделением интентов
    """
    greeting_examples = [
        "привет", "здравствуй", "добрый день", "доброе утро", "добрый вечер",
        "хай", "здарова", "приветствую", "здравствуйте", "доброго времени суток",
        "салют", "приветик", "здрасьте", "здорово", "привет всем",
        "здравствуйте всем", "вечер в хату", "всем привет", "приветствую вас",
        "здорово братан", "хаюшки", "здравствуйте как жизнь", "добрый вечерок"
    ]
    
    weather_examples = [
        "какая погода", "погода в москве", "сколько градусов", "температура на улице",
        "будет ли дождь", "прогноз погоды", "что с погодой", "холодно на улице",
        "жарко сегодня", "какая температура завтра", "погода на завтра", "снег будет",
        "ветер сильный", "прогноз на сегодня", "метео", "погода сейчас", "температура сейчас",
        "какой прогноз", "что за погода", "тепло ли на улице", "градусов сколько",
        "на улице холодно", "на улице жарко", "погодные условия"
    ]
    
    goodbye_examples = [
        "пока", "до свидания", "всего хорошего", "до встречи", "увидимся",
        "прощай", "бывай", "счастливо", "удачи", "до скорого",
        "всего доброго", "покеда", "чао", "до связи", "пока пока",
        "прощайте", "до завтра", "увидимся позже", "до вечера", "до скорой встречи",
        "пока удачи", "счастливо оставаться", "бывайте", "всего наилучшего"
    ]
    
    addition_examples = [
        "сложи 5 и 3", "2 плюс 2", "сумма 10 и 20", "прибавь 15 к 7",
        "сколько будет 8 + 4", "сложение", "плюс", "прибавить",
        "сложи числа 12 и 8", "вычисли сумму 3 и 6", "плюс 5 и 5",
        "сложить 4 и 9", "прибавь 10", "сумма 1 и 1", "калькулятор",
        "сложи 100 и 200", "прибавь 5", "плюс 7", "добавь 3", "плюс 8"
    ]
    
    set_name_examples = [
        "меня зовут Максим", "меня зовут Анна", "меня зовут Вовка", "меня зовут Карл",
        "меня зовут Антон", "меня зовут Петя", "меня зовут Маша", "меня зовут Саша",
        "меня зовут Дима", "меня зовут Лена", "меня зовут Олег", "меня зовут Катя",
        "мое имя Дмитрий", "мое имя Иван", "мое имя Сергей", "мое имя Алексей",
        "мое имя Владимир", "мое имя Наташа", "мое имя Ольга", "мое имя Татьяна",
        "называй меня Саша", "называй меня Дима", "называй меня Костя", "называй меня Петя",
        "называй меня Макс", "называй меня Жорик", "называй меня Боб", "называй меня Джон",
        "зови меня Катя", "зови меня Олег", "зови меня Андрей", "зови меня Денис",
        "зови меня Скам", "зови меня Том", "зови меня Джек", "зови меня Макс",
        "я Максим", "я Анна", "я Вовка", "я Карл", "я Антон", "я Петя", "я Маша",
        "меня звать Олег", "меня звать Андрей", "меня звать Денис", "меня звать Юля"
    ]
    
    unknown_examples = [
        "как дела", "что ты умеешь", "расскажи шутку", "кто твой создатель",
        "сколько времени", "какой сегодня день", "помоги мне", "спасибо",
        "что нового", "как тебя зовут", "откуда ты", "ты человек",
        "расскажи о себе", "как настроение", "что делать", "зачем ты нужен",
        "кто ты", "сколько тебе лет", "где ты живешь", "какие новости"
    ]
    
    texts = []
    texts.extend(greeting_examples)
    texts.extend(weather_examples)
    texts.extend(goodbye_examples)
    texts.extend(addition_examples)
    texts.extend(set_name_examples)
    texts.extend(unknown_examples)
    
    intents = []
    intents.extend(['greeting'] * len(greeting_examples))
    intents.extend(['weather'] * len(weather_examples))
    intents.extend(['goodbye'] * len(goodbye_examples))
    intents.extend(['addition'] * len(addition_examples))
    intents.extend(['set_name'] * len(set_name_examples))
    intents.extend(['unknown'] * len(unknown_examples))
    
    df = pd.DataFrame({'text': texts, 'intent': intents})
    
    df.to_csv("dataset.csv", index=False, encoding='utf-8')
    print(f"Датасет создан: {len(df)} примеров")
    print(f"Распределение по интентам:\n{df['intent'].value_counts()}")
    
    return df

def train_model_embeddings():
    """
    Обучение модели классификации интентов с использованием Word Embeddings (spaCy)
    """
    print("=" * 70)
    print("Обучение ML-модели для классификации интентов")
    print("Метод: Word Embeddings (spaCy) + LogisticRegression")
    print("=" * 70)
    
    if not os.path.exists("dataset.csv"):
        print("Создание датасета...")
        df = create_dataset()
    else:
        df = pd.read_csv("dataset.csv")
        print(f"Загружен датасет: {len(df)} примеров")
        print(f"Распределение по интентам:\n{df['intent'].value_counts()}")
    
    texts = df["text"].tolist()
    labels = df["intent"].tolist()
    
    print("\nШаг 1: Предобработка текстов (лемматизация, удаление стоп-слов)...")
    processed_texts = [preprocess_text(text) for text in texts]
    
    print("\nШаг 2: Получение векторных представлений через Word Embeddings...")
    print("Используется: усреднение векторов слов с помощью spaCy")
    
    X = np.array([get_word_embedding(text) for text in processed_texts])
    
    print(f"\nРазмерность признакового пространства: {X.shape[1]}")
    print(f"Количество примеров: {X.shape[0]}")
    print(f"Форма матрицы признаков: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nОбучающая выборка: {X_train.shape[0]} примеров")
    print(f"Тестовая выборка: {X_test.shape[0]} примеров")
    
    print("\nШаг 3: Обучение модели LogisticRegression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nТочность модели: {accuracy:.2%}")
    print("\nДетальный отчет по классам:")
    print(classification_report(y_test, y_pred))
    
    print("\nШаг 4: Сохранение модели...")
    joblib.dump(model, "intent_model_embeddings.pkl")
    print("Модель сохранена в 'intent_model_embeddings.pkl'")
    
    print("\n" + "=" * 70)
    print("Тестирование модели на примерах:")
    print("=" * 70)
    
    test_examples = [
        "привет",
        "меня зовут Антон",
        "меня зовут Вовка",
        "называй меня Макс",
        "зови меня Жорик",
        "я Петя",
        "мое имя Олег",
        "погода в москве",
        "сложи 10 и 5",
        "2 плюс 2",
        "до свидания",
        "пока",
        "как дела",
        "что ты умеешь"
    ]
    
    for example in test_examples:
        processed = preprocess_text(example)
        vec = get_word_embedding(processed).reshape(1, -1)
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)
        confidence = max(prob[0])
        
        print(f"\nИсходный текст: '{example}'")
        print(f"  Предобработка: '{processed[:50]}...'")
        print(f"  Предсказание: {pred} (уверенность: {confidence:.2%})")
    
    return model

if __name__ == "__main__":
    train_model_embeddings()