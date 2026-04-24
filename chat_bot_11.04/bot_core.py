from bert_intent import load_bert_model, predict_intent_bert
import re
from datetime import datetime
import sqlite3
from weather_api import get_weather
from enum import Enum
import json
import os

class BotState(Enum):
    START = "start"
    WAITING_CITY = "waiting_city"
    WAITING_DATE = "waiting_date"
    WAITING_FIRST_NUMBER = "waiting_first_number"
    WAITING_SECOND_NUMBER = "waiting_second_number"
    WAITING_CONFIRMATION = "waiting_confirmation"
    WAITING_FOLLOW_UP = "waiting_follow_up"

conversation_history = {}
user_last_response = {}

def load_bert():
    return load_bert_model()

def is_bert_ready():
    from bert_intent import is_bert_available
    return is_bert_available()

def init_db():
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
            user_id INTEGER,bot
            user_query TEXT,
            city TEXT,
            weather_response TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dialog_states (
            user_id INTEGER PRIMARY KEY,
            state TEXT,
            temp_data TEXT,
            last_updated TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bert_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_text TEXT,
            predicted_intent TEXT,
            confidence REAL,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    conn.commit()
    conn.close()

def save_user(user_id, name):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO users (user_id, name, last_interaction) VALUES (?, ?, ?)",
        (user_id, name, datetime.now())
    )
    conn.commit()
    conn.close()

def get_user(user_id):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_weather_query(user_id, user_query, city, weather_response):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO weather_queries (user_id, user_query, city, weather_response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, user_query, city, weather_response, datetime.now()))
    conn.commit()
    conn.close()

def save_bert_prediction(user_id, text, intent, confidence):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO bert_predictions (user_id, original_text, predicted_intent, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, text, intent, confidence, datetime.now()))
    conn.commit()
    conn.close()

def save_dialog_state(user_id, state, temp_data=None):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    temp_data_json = json.dumps(temp_data) if temp_data else "{}"
    cursor.execute("""
        INSERT OR REPLACE INTO dialog_states (user_id, state, temp_data, last_updated)
        VALUES (?, ?, ?, ?)
    """, (user_id, state.value if isinstance(state, BotState) else state, temp_data_json, datetime.now()))
    conn.commit()
    conn.close()

def load_dialog_state(user_id):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT state, temp_data FROM dialog_states WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        state_str, temp_data_json = result
        try:
            state = BotState(state_str)
        except ValueError:
            state = BotState.START
        temp_data = json.loads(temp_data_json) if temp_data_json else {}
        return state, temp_data
    return BotState.START, {}

def extract_city_from_text(text):
    text_lower = text.lower()
    common_cities = ['москв', 'спб', 'питер', 'новосибирск', 'екатеринбург', 
                     'казан', 'нижний', 'челябинск', 'омск', 'самар', 'ростов']
    for city in common_cities:
        if city in text_lower:
            if city == 'москв':
                return 'Москва'
            elif city == 'спб' or city == 'питер':
                return 'Санкт-Петербург'
            elif city == 'казан':
                return 'Казань'
            elif city == 'нижний':
                return 'Нижний Новгород'
            elif city == 'самар':
                return 'Самара'
            elif city == 'ростов':
                return 'Ростов-на-Дону'
            else:
                return city.capitalize()
    return None

def extract_date_from_text(text):
    text_lower = text.lower()
    if 'сегодня' in text_lower or 'сейчас' in text_lower:
        return 'сегодня'
    elif 'завтра' in text_lower:
        return 'завтра'
    elif 'послезавтра' in text_lower:
        return 'послезавтра'
    return None

def handle_farewell():
    return "До свидания! Будет приятно видеть вас снова."

def log_message(user_message, bot_response):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] USER: {user_message}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BOT: {bot_response}\n")
        f.write("-" * 50 + "\n")

def process_confirmation(message: str, bot_instance: 'ChatBot') -> str:
    if message.lower() in ['да', 'конечно', 'ага', 'ок', 'yes', '+', 'давай', 'хорошо']:
        bot_instance.reset_state()
        return "Отлично! Что дальше?"
    elif message.lower() in ['нет', 'не надо', 'отмена', 'no', 'не', 'неа']:
        bot_instance.reset_state()
        return "Хорошо, отменяю."
    else:
        return "Пожалуйста, ответьте 'да' или 'нет'."

def process_follow_up(message: str, bot_instance: 'ChatBot') -> str:
    history = conversation_history.get(bot_instance.user_id, [])
    if len(history) >= 1:
        last_intent = history[-1][2]
        if last_intent == "weather":
            city = extract_city_from_text(message)
            if city:
                return bot_instance.process_city(message)
            else:
                return "Уточните, пожалуйста, город."
        elif last_intent == "addition":
            numbers = re.findall(r"[-+]?\d*\.?\d+", message.replace(',', '.'))
            if numbers:
                return bot_instance.start_addition_dialog(message)
            else:
                return "Пожалуйста, введите число для сложения."
    bot_instance.reset_state()
    return "Давайте начнём сначала. Что вы хотите?"

class ChatBot:
    def __init__(self, user_id=None):
        self.name = None
        self.user_id = user_id
        self.state = BotState.START
        self.temp_data = {}
        if user_id:
            self.name = get_user(user_id)
            self.state, self.temp_data = load_dialog_state(user_id)
    
    def save_state(self):
        if self.user_id:
            save_dialog_state(self.user_id, self.state, self.temp_data)
    
    def reset_state(self):
        self.state = BotState.START
        self.temp_data = {}
        self.save_state()
    
    def set_name(self, name):
        self.name = name
        if self.user_id and self.name:
            save_user(self.user_id, self.name)
        self.reset_state()    
        return f"Приятно познакомиться, {self.name}! Я запомнил ваше имя."

    def greet(self):
        if self.name:
            return f"Здравствуйте, {self.name}! Рад вас снова видеть."
        return "Здравствуйте! Как я могу к вам обращаться?"

    def start_weather_dialog(self, message):
        city = extract_city_from_text(message)
        date = extract_date_from_text(message)
        self.temp_data = {'intent': 'weather'}
        if city:
            self.temp_data['city'] = city
            if date:
                weather_response = get_weather(city)
                if self.user_id:
                    save_weather_query(self.user_id, message, city, weather_response)
                self.reset_state()
                return f"{weather_response}\n\nЧем ещё могу помочь?"
            else:
                self.state = BotState.WAITING_DATE
                self.save_state()
                return f"Город {city} принят. На какую дату нужен прогноз? (сегодня/завтра)"
        else:
            self.state = BotState.WAITING_CITY
            self.save_state()
            return "В каком городе вы хотите узнать погоду?"

    def start_addition_dialog(self, message):
        numbers = re.findall(r"[-+]?\d*\.?\d+", message.replace(',', '.'))
        if len(numbers) >= 2:
            try:
                a = float(numbers[0])
                b = float(numbers[1])
                result = a + b
                self.reset_state()
                return f"Результат: {a} + {b} = {result}"
            except ValueError:
                pass
        self.temp_data = {'intent': 'addition'}
        self.state = BotState.WAITING_FIRST_NUMBER
        self.save_state()
        return "Введите первое число:"

    def process_city(self, city_text):
        city = extract_city_from_text(city_text)
        if not city:
            city = city_text.strip()
        if not city:
            return "Пожалуйста, укажите название города."
        self.temp_data['city'] = city
        date = extract_date_from_text(city_text)
        if date:
            weather_response = get_weather(city)
            if self.user_id:
                save_weather_query(self.user_id, city_text, city, weather_response)
            self.reset_state()
            return f"{weather_response}\n\nЧем ещё могу помочь?"
        self.state = BotState.WAITING_DATE
        self.save_state()
        return f"Город {city} принят. На какую дату нужен прогноз? (сегодня/завтра)"

    def process_date(self, date_text):
        date = extract_date_from_text(date_text)
        if not date:
            date = 'сегодня'
        self.temp_data['date'] = date
        city = self.temp_data.get('city')
        weather_response = get_weather(city)
        if self.user_id:
            save_weather_query(self.user_id, f"погода в {city} на {date}", city, weather_response)
        self.reset_state()
        return f"{weather_response}\n\nЧем ещё могу помочь?"

    def process_first_number(self, number_text):
        try:
            numbers = re.findall(r"[-+]?\d*\.?\d+", number_text.replace(',', '.'))
            if numbers:
                number = float(numbers[0])
            else:
                number = float(number_text.replace(',', '.'))
            self.temp_data['first_number'] = number
            if len(numbers) >= 2:
                second = float(numbers[1])
                result = number + second
                self.reset_state()
                return f"Результат: {number} + {second} = {result}"
            self.state = BotState.WAITING_SECOND_NUMBER
            self.save_state()
            return f"Первое число: {number}. Введите второе число:"
        except ValueError:
            return "Пожалуйста, введите корректное число."

    def process_second_number(self, number_text):
        try:
            numbers = re.findall(r"[-+]?\d*\.?\d+", number_text.replace(',', '.'))
            if numbers:
                second = float(numbers[0])
            else:
                second = float(number_text.replace(',', '.'))
            first = self.temp_data.get('first_number', 0)
            result = first + second
            self.reset_state()
            return f"Результат: {first} + {second} = {result}"
        except ValueError:
            return "Пожалуйста, введите корректное число."

    def get_time(self) -> str:
        return f" Сейчас {datetime.now().strftime('%H:%M:%S')}"
    
    def get_date(self) -> str:
        return f" Сегодня {datetime.now().strftime('%d.%m.%Y')}"
    
    def get_help(self) -> str:
        return """ **Я умею:**

 **Погода** - "какая погода в Москве"
 **Сложение** - "сложи 5 и 3" или "2 плюс 2"
 **Запоминать имя** - "меня зовут Антон"
 **Время** - "сколько времени"
 **Дата** - "какое сегодня число"
 **Справка** - "что ты умеешь"
 **Благодарность** - "спасибо"
 **Повтор** - "повтори"
 **Отмена** - "отмена"

Также я поддерживаю диалог в несколько шагов!"""
    
    def handle_thanks(self) -> str:
        return "Пожалуйста! Обращайтесь ещё "
    
    def handle_repeat(self) -> str:
        last_resp = user_last_response.get(self.user_id, "")
        if last_resp:
            return f" Повторяю: {last_resp}"
        return "У меня пока нет последнего ответа для повтора."
    
    def handle_cancel(self) -> str:
        self.reset_state()
        return " Действие отменено. Чем могу помочь?"

def process_message(message: str, bot_instance: ChatBot) -> str:
    message = message.strip()
    if not message:
        return "Вы ничего не написали. Чем могу помочь?"
    try:
        if bot_instance.state == BotState.WAITING_CITY:
            return bot_instance.process_city(message)
        elif bot_instance.state == BotState.WAITING_DATE:
            return bot_instance.process_date(message)
        elif bot_instance.state == BotState.WAITING_FIRST_NUMBER:
            return bot_instance.process_first_number(message)
        elif bot_instance.state == BotState.WAITING_SECOND_NUMBER:
            return bot_instance.process_second_number(message)
        elif bot_instance.state == BotState.WAITING_CONFIRMATION:
            return process_confirmation(message, bot_instance)
        elif bot_instance.state == BotState.WAITING_FOLLOW_UP:
            return process_follow_up(message, bot_instance)
        
        intent, confidence = predict_intent_bert(message)
        if not intent:
            return "Извините, BERT модель не загружена."
        print(f"[BERT] Интент: {intent}, уверенность: {confidence:.2%}")
        if bot_instance.user_id and intent:
            save_bert_prediction(bot_instance.user_id, message, intent, confidence)
        if confidence < 0.4:
            return "Извините, я не уверен, что правильно понял ваш запрос."
        
        if intent == "greeting":
            response = bot_instance.greet()
        elif intent == "goodbye":
            response = handle_farewell()
        elif intent == "weather":
            response = bot_instance.start_weather_dialog(message)
        elif intent == "addition":
            response = bot_instance.start_addition_dialog(message)
        elif intent == "set_name":
            name_match = re.search(r"(?:меня зовут|мое имя|называй меня|зови меня|меня звать|я)\s+([а-яА-Яa-zA-Z]+)", message, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
                response = bot_instance.set_name(name)
            else:
                response = "Как вас зовут? Скажите, например: 'Меня зовут Анна'"
        elif intent == "time":
            response = bot_instance.get_time()
        elif intent == "date":
            response = bot_instance.get_date()
        elif intent == "help":
            response = bot_instance.get_help()
        elif intent == "thanks":
            response = bot_instance.handle_thanks()
        elif intent == "repeat":
            response = bot_instance.handle_repeat()
        elif intent == "cancel":
            response = bot_instance.handle_cancel()
        elif intent == "how_are_you":
            if bot_instance.name:
                response = f"У меня всё отлично, {bot_instance.name}! Спасибо, что спросили:)) А как ваши дела?"
            else:
                response = "У меня всё отлично! Спасибо, что спросили:)) А как ваши дела?"
        elif intent == "unknown":
            response = "Я не совсем понял ваш запрос. Скажите 'помощь' чтобы узнать что я умею."
        else:
            response = "Я не понимаю запрос. Попробуйте переформулировать."
        
        if bot_instance.user_id:
            if bot_instance.user_id not in conversation_history:
                conversation_history[bot_instance.user_id] = []
            conversation_history[bot_instance.user_id].append((message, response, intent))
            if len(conversation_history[bot_instance.user_id]) > 10:
                conversation_history[bot_instance.user_id].pop(0)
            user_last_response[bot_instance.user_id] = response
        return response
    except Exception as e:
        print(f"Ошибка: {e}")
        return "Произошла ошибка. Попробуйте ещё раз."