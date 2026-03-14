import re
from datetime import datetime
import sqlite3
from weather_api import get_weather
import spacy
from enum import Enum
from typing import Dict, Any, Optional

nlp = spacy.load("ru_core_news_sm")

class DialogState(Enum):
    START = "start"
    WAIT_CITY = "wait_city"
    WAIT_DATE = "wait_date"
    WAIT_CONFIRMATION = "wait_confirmation"

user_states: Dict[int, Dict[str, Any]] = {}

def get_state(user_id: int) -> DialogState:
    """Получение текущего состояния диалога пользователя"""
    if user_id in user_states and "state" in user_states[user_id]:
        return user_states[user_id]["state"]
    return DialogState.START

def set_state(user_id: int, state: DialogState):
    """Установка состояния диалога пользователя"""
    if user_id not in user_states:
        user_states[user_id] = {}
    user_states[user_id]["state"] = state

def get_user_data(user_id: int) -> Dict[str, Any]:
    """Получение данных пользователя"""
    if user_id not in user_states:
        user_states[user_id] = {}
    return user_states[user_id]

def clear_user_data(user_id: int):
    """Очистка данных пользователя"""
    if user_id in user_states:
        user_states[user_id] = {"state": DialogState.START}

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
            weather_date TEXT,
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

def save_weather_query(user_id, user_query, city, weather_response, weather_date=None):
    """Сохранение запроса погоды в базу данных"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO weather_queries (user_id, user_query, city, weather_date, weather_response, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, user_query, city, weather_date, weather_response, datetime.now()))

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

def extract_date_with_spacy(text):
    """
    Извлекает дату из текста с помощью spaCy
    """
    doc = nlp(text)
    
    date_keywords = ["сегодня", "завтра", "послезавтра", "понедельник", "вторник", 
                     "среда", "четверг", "пятница", "суббота", "воскресенье"]
    
    text_lower = text.lower()
    for keyword in date_keywords:
        if keyword in text_lower:
            return keyword
    
    date_pattern = re.compile(r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b", re.IGNORECASE)
    match = date_pattern.search(text)
    if match:
        return match.group(0)
    
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
    """
    Анализирует текст с помощью spaCy
    """
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
            if user_id not in user_states:
                user_states[user_id] = {"state": DialogState.START}

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

class DialogManager:
    """Менеджер диалога на основе конечного автомата"""
    
    def __init__(self, bot_instance: ChatBot):
        self.bot = bot_instance
    
    def process_message(self, text: str) -> str:
        """Обработка сообщения с учётом состояния диалога"""
        
        if not self.bot.user_id:
            return self._process_without_state(text)
        
        current_state = get_state(self.bot.user_id)
        user_data = get_user_data(self.bot.user_id)
        
        analysis = analyze_text_with_spacy(text)
        if self.bot.user_id:
            save_text_analysis(self.bot.user_id, text, analysis)
        
        if any(word in text.lower() for word in ['отмена', 'выйти', 'забудь', 'не надо']):
            set_state(self.bot.user_id, DialogState.START)
            clear_user_data(self.bot.user_id)
            return "Диалог отменён. Чем ещё могу помочь?"
        
        if current_state == DialogState.START:
            return self._handle_start_state(text)
        
        elif current_state == DialogState.WAIT_CITY:
            return self._handle_wait_city_state(text, user_data)
        
        elif current_state == DialogState.WAIT_DATE:
            return self._handle_wait_date_state(text, user_data)
        
        elif current_state == DialogState.WAIT_CONFIRMATION:
            return self._handle_wait_confirmation_state(text, user_data)
        
        return "Извините, произошла ошибка. Начнём сначала."
    
    def _process_without_state(self, text: str) -> str:
        """Обработка без сохранения состояния (для тестов)"""
        if is_weather_query_with_spacy(text):
            city = extract_city_with_spacy(text)
            if city:
                return get_weather(city)
            else:
                return "Пожалуйста, укажите город в запросе о погоде. Например: 'погода в Москве'"
        return "Я не понимаю запрос. Попробуйте спросить о погоде."
    
    def _handle_start_state(self, text: str) -> str:
        """Обработка в начальном состоянии"""
        
        if is_weather_query_with_spacy(text):
            city = extract_city_with_spacy(text)
            
            if city:
                date = extract_date_with_spacy(text)
                
                if date and date not in ["сегодня", "завтра"]:
                    user_data = get_user_data(self.bot.user_id)
                    user_data["city"] = city
                    user_data["date"] = date
                    set_state(self.bot.user_id, DialogState.WAIT_CONFIRMATION)
                    
                    return f"Вы хотите узнать погоду в городе {city} на {date}? (да/нет)"
                else:
                    weather_response = get_weather(city)
                    if self.bot.user_id:
                        save_weather_query(self.bot.user_id, text, city, weather_response)
                    
                    return weather_response
            else:
                set_state(self.bot.user_id, DialogState.WAIT_CITY)
                return "В каком городе вас интересует погода?"
        
        greeting_pattern = re.compile(r"^(привет|здравствуй|добрый день|доброе утро|добрый вечер|хай|здарова)$", re.IGNORECASE)
        if greeting_pattern.match(text):
            return self.bot.greet()
        
        farewell_pattern = re.compile(r"^(пока|до свидания|всего хорошего|до встречи|увидимся)$", re.IGNORECASE)
        if farewell_pattern.match(text):
            return handle_farewell()
        
        addition_pattern = re.compile(r"(\d+)\s*\+\s*(\d+)")
        match = addition_pattern.search(text)
        if match:
            return handle_addition(match)
        
        name_pattern = re.compile(r"(?:меня зовут|мое имя|называй меня|я) ([а-яА-Яa-zA-Z]+)", re.IGNORECASE)
        match = name_pattern.search(text)
        if match:
            return self.bot.set_name(match)
        
        return "Я не понимаю запрос. Попробуйте спросить о погоде (например: 'Какая погода в Москве?')"
    
    def _handle_wait_city_state(self, text: str, user_data: Dict) -> str:
        """Обработка в состоянии ожидания города"""
        
        city = extract_city_with_spacy(text)
        
        if not city:
            words = text.split()
            if len(words) > 0:
                city_candidate = words[-1].strip().capitalize()
                if len(city_candidate) > 2 and city_candidate.lower() not in ["погода", "город", "городе"]:
                    city = city_candidate
        
        if city:
            date = extract_date_with_spacy(text)
            
            if date and date not in ["сегодня", "завтра"]:
                user_data["city"] = city
                user_data["date"] = date
                set_state(self.bot.user_id, DialogState.WAIT_CONFIRMATION)
                return f"Вы хотите узнать погоду в городе {city} на {date}? (да/нет)"
            else:
                weather_response = get_weather(city)
                if self.bot.user_id:
                    save_weather_query(self.bot.user_id, text, city, weather_response)
                
                set_state(self.bot.user_id, DialogState.START)
                return weather_response
        else:
            return "Я не смог распознать город. Пожалуйста, укажите название города ещё раз."
    
    def _handle_wait_date_state(self, text: str, user_data: Dict) -> str:
        """Обработка в состоянии ожидания даты"""
        
        date = extract_date_with_spacy(text)
        
        if date:
            city = user_data.get("city")
            if city:
                weather_response = get_weather(city)
                if self.bot.user_id:
                    save_weather_query(self.bot.user_id, f"погода в {city} на {date}", city, weather_response, date)
                
                set_state(self.bot.user_id, DialogState.START)
                return f"Прогноз погоды в городе {city} на {date}:\n{weather_response}"
            else:
                set_state(self.bot.user_id, DialogState.WAIT_CITY)
                return "Не удалось определить город. В каком городе вас интересует погода?"
        else:
            return "Пожалуйста, укажите дату (например: 'завтра', 'послезавтра' или '15 мая')"
    
    def _handle_wait_confirmation_state(self, text: str, user_data: Dict) -> str:
        """Обработка в состоянии ожидания подтверждения"""
        
        text_lower = text.lower()
        
        if text_lower in ["да", "давай", "ок", "хорошо", "ага", "yes", "y"]:
            city = user_data.get("city")
            date = user_data.get("date", "сегодня")
            
            if city:
                weather_response = get_weather(city)
                if self.bot.user_id:
                    save_weather_query(self.bot.user_id, f"погода в {city} на {date}", city, weather_response, date)
                
                set_state(self.bot.user_id, DialogState.START)
                return f"Прогноз погоды в городе {city} на {date}:\n{weather_response}"
            else:
                set_state(self.bot.user_id, DialogState.WAIT_CITY)
                return "В каком городе вас интересует погода?"
        
        elif text_lower in ["нет", "не надо", "отмена", "no", "n"]:
            set_state(self.bot.user_id, DialogState.START)
            return "Хорошо. Чем ещё могу помочь?"
        
        else:
            return "Пожалуйста, ответьте 'да' или 'нет'."

def create_patterns(bot_instance):
    """Создание списка паттернов с обработчиками (для обратной совместимости)"""
    return [
        (re.compile(r"^(привет|здравствуй|добрый день|доброе утро|добрый вечер|хай|здарова)$", re.IGNORECASE), 
         bot_instance.greet),
        
        (re.compile(r"^(пока|до свидания|всего хорошего|до встречи|увидимся)$", re.IGNORECASE), 
         handle_farewell),
        
        (re.compile(r"(\d+)\s*\+\s*(\d+)"), 
         handle_addition),
        
        (re.compile(r"(?:меня зовут|мое имя|называй меня|я) ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), 
         bot_instance.set_name),
    ]

def process_message(message: str, bot_instance: ChatBot) -> str:
    """
    Обработка сообщения пользователя с использованием Dialog Manager
    """
    message = message.strip()
    
    if not message:
        return "Вы ничего не написали. Чем могу помочь?"

    try:
        dialog_manager = DialogManager(bot_instance)
        response = dialog_manager.process_message(message)
        
        return response
    
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
        
        return "Я не понимаю запрос. Попробуйте: 'погода в Москве'"