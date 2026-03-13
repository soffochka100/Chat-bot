import re
from datetime import datetime
import sqlite3
from weather_api import get_weather
import spacy


nlp = spacy.load("ru_core_news_sm")

def init_db():
    """Создание таблиц пользователей и запросов погоды"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            last_interaction TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_query TEXT,
            city TEXT,
            weather_response TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_text TEXT,
            tokens TEXT,
            lemmas TEXT,
            pos_tags TEXT,
            entities TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    conn.commit()
    conn.close()

def save_user(user_id, name):
    """Сохранение пользователя в БД"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR REPLACE INTO users (user_id, name, last_interaction) VALUES (?, ?, ?)",
        (user_id, name, datetime.now())
    )

    conn.commit()
    conn.close()

def get_user(user_id):
    """Получение имени пользователя из БД"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None

def save_weather_query(user_id, user_query, city, weather_response):
    """Сохранение запроса погоды в базу данных"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO weather_queries (user_id, user_query, city, weather_response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, user_query, city, weather_response, datetime.now()))

    conn.commit()
    conn.close()

def save_text_analysis(user_id, text, analysis):
    """Сохраняет результаты анализа текста в базу данных"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO text_analysis 
        (user_id, original_text, tokens, lemmas, pos_tags, entities, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, 
        text,
        ', '.join(analysis['tokens']),
        ', '.join(analysis['lemmas']),
        ', '.join(analysis['pos_tags']),
        ', '.join([f"{ent}({label})" for ent, label in analysis['entities']]),
        datetime.now()
    ))

    conn.commit()
    conn.close()

def extract_city_with_spacy(text):
    """
    Извлекает название города из текста с помощью spaCy NER
    Поддерживает разные падежи (Москва, Москве, Москвой, Мокве и т.д.)
    """
    doc = nlp(text)
    cities = []
    
    for ent in doc.ents:
        
        
        if ent.label_ in ["GPE", "LOC"]:
            
            
            return ent.lemma_
    
    return None

def is_weather_query_with_spacy(text):
    """
    Анализирует, является ли запрос запросом погоды, используя NLP
    """
    text_lower = text.lower()
    
    weather_keywords = [
        "погод", "температур", "прогноз", "жарк", "холодн",
        "дожд", "снег", "ветер", "солнечн", "облачн", "град",
        "тепл", "мороз", "осадк", "метео"
    ]
    
    for keyword in weather_keywords:
        if keyword in text_lower:
            return True
    
    return False

def analyze_text_with_spacy(text):
   
    doc = nlp(text)
    
    analysis = {
        "original": text,
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "pos_tags": [token.pos_ for token in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }
    
    return analysis

def handle_greeting(match=None):
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell(match=None):
    return "До свидания! Будет приятно видеть вас снова."

def handle_weather_with_spacy(match, bot_instance, user_text):
    
    city = extract_city_with_spacy(user_text)
    
    if not city:
        weather_pattern = re.compile(r"(?:в|во|на|из|около|возле|под|про)\s+([а-яА-Яa-zA-Z\-]+)", re.IGNORECASE)
        match = weather_pattern.search(user_text)
        if match:
            city_candidate = match.group(1)
            try:
                doc = nlp(city_candidate)
                if doc and len(doc) > 0:
                    
                    city = doc[0].lemma_
                else:
                    city = city_candidate
            except:
                city = city_candidate
    
    if not city:
        return "Пожалуйста, укажите город в запросе о погоде. Например: 'погода в Москве' или 'погода в Питере'"
    
    weather_response = get_weather(city)
    
    if bot_instance.user_id:
        save_weather_query(bot_instance.user_id, user_text, city, weather_response)
    
    return weather_response

def handle_addition(match):
    """Сложение двух чисел"""
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a} + {b} = {a + b}"

def log_message(user_message, bot_response):
    """Логирование сообщений в файл"""
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] USER: {user_message}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BOT: {bot_response}\n")
        f.write("-" * 50 + "\n")

class ChatBot:
    def __init__(self, user_id=None):
        self.name = None
        self.user_id = user_id
        
        if user_id:
            self.name = get_user(user_id)

    def set_name(self, match):
        """Установка имени пользователя"""
        self.name = match.group(1)
        
        if self.user_id and self.name:
            save_user(self.user_id, self.name)
            
        return f"Приятно познакомиться, {self.name}! Я запомнил ваше имя."

    def greet(self, match=None):
        """Приветствие с учётом имени"""
        if self.name:
            return f"Здравствуйте, {self.name}! Рад вас снова видеть."
        return "Здравствуйте! Как я могу к вам обращаться?"

def create_patterns(bot_instance):
    """Создание списка паттернов с обработчиками"""
    return [
        (re.compile(r"^(привет|здравствуй|добрый день|доброе утро|добрый вечер|хай|здарова)$", re.IGNORECASE), 
         bot_instance.greet),
        
        (re.compile(r"^(пока|до свидания|всего хорошего|до встречи|увидимся)$", re.IGNORECASE), 
         handle_farewell),
        
        (re.compile(r".*(?:погода|температура|прогноз|холодно|жарко|дождь|снег).*(?:в|во|на|из|около|под|про)\s+([а-яА-Яa-zA-Z\-]+)", re.IGNORECASE), 
         lambda m: handle_weather_with_spacy(m, bot_instance, m.string)),
        
        (re.compile(r"^погода\s+([а-яА-Яa-zA-Z\-]+)$", re.IGNORECASE), 
         lambda m: handle_weather_with_spacy(m, bot_instance, f"погода в {m.group(1)}")),
        
        (re.compile(r"(\d+)\s*\+\s*(\d+)"), 
         handle_addition),
        
        (re.compile(r"(?:меня зовут|мое имя|называй меня|я) ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), 
         bot_instance.set_name),
    ]

def process_message(message: str, bot_instance: ChatBot) -> str:
    """
    Обработка сообщения пользователя с поддержкой разных падежей
    """
    message = message.strip()
    
    if not message:
        return "Вы ничего не написали. Чем могу помочь?"

    try:
        analysis = analyze_text_with_spacy(message)
        
        if bot_instance.user_id:
            save_text_analysis(bot_instance.user_id, message, analysis)

        if is_weather_query_with_spacy(message):
            city = extract_city_with_spacy(message)
            
            if city:
                weather_response = get_weather(city)
                
                if bot_instance.user_id:
                    save_weather_query(bot_instance.user_id, message, city, weather_response)
                
                return weather_response
            else:
                
                patterns = create_patterns(bot_instance)
                for pattern, handler in patterns:
                    if "погод" in pattern.pattern.lower():
                        match = pattern.search(message)
                        if match:
                            return handler(match)

        
        patterns = create_patterns(bot_instance)

        for pattern, handler in patterns:
            match = pattern.search(message)
            if match:
                return handler(match)

        return "Я не понимаю запрос. Попробуйте спросить о погоде (например: 'Какая погода в Москве?' или 'погода в Питере')"
    
    except Exception as e:
        print(f"Ошибка: {e}")
        
        
        if "погода" in message.lower():
            parts = message.lower().split()
            for i, word in enumerate(parts):
                if word in ["в", "во"] and i+1 < len(parts):
                    city = parts[i+1]
                    return get_weather(city)
            
            words = message.split()
            if len(words) > 0:
                city = words[-1]
                return get_weather(city)
        
        patterns = create_patterns(bot_instance)
        for pattern, handler in patterns:
            match = pattern.search(message)
            if match:
                return handler(match)
        
        return "Я не понимаю запрос. Попробуйте: 'погода в Москве'"