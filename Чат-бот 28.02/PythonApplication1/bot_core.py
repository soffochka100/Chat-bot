import re
import string
import sqlite3
from datetime import datetime
from weather_api import get_weather

DB_NAME = "bot.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_user(name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (name) VALUES (?)
        ON CONFLICT(name) DO UPDATE SET first_seen = CURRENT_TIMESTAMP
    """, (name,))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id

def get_user_by_name(name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name, first_seen FROM users WHERE name = ?", (name,))
    result = cursor.fetchone()
    conn.close()
    return result

def handle_greeting():
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell():
    return "До свидания!"

def handle_weather(match):
    city = match.group(1).strip()
    return get_weather(city)

def handle_time(match=None):
    current_time = datetime.now().strftime("%H:%M")
    current_date = datetime.now().strftime("%d.%m.%Y")
    return f"Сегодня {current_date}, время {current_time}"

def handle_addition(match):
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a + b}"

def handle_subtraction(match):
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a - b}"

def handle_set_name(match):
    name = match.group(1).strip()
    user_id = save_user(name)
    return f"Приятно познакомиться, {name}!"

def log_message(user, bot):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] USER: {user}\n")
        f.write(f"[{datetime.now()}] BOT: {bot}\n")
        f.write("-" * 50 + "\n")

class ChatBot:
    def __init__(self):
        self.name = None
        self.patterns = []
        self.register_patterns()

    def register_patterns(self):
        self.patterns.append(
            (re.compile(r"привет", re.IGNORECASE), self.greet)
        )
        self.patterns.append(
            (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), self.set_name)
        )

    def greet(self, match):
        if self.name:
            return f"Здравствуйте, {self.name}!"
        return "Здравствуйте!"

    def set_name(self, match):
        self.name = match.group(1)
        save_user(self.name)
        return f"Приятно познакомиться, {self.name}!"

    def process(self, message):
        message = message.lower().strip()
        message = message.translate(str.maketrans('', '', string.punctuation))
        
        for pattern, handler in self.patterns:
            match = pattern.search(message)
            if match:
                return handler(match)
        return None

patterns = [
    (re.compile(r"^(привет|здравствуй|добрый день|добрый вечер|доброе утро)$", re.IGNORECASE), handle_greeting),
    (re.compile(r"^(пока|до свидания|всего хорошего)$", re.IGNORECASE), handle_farewell),
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\-\s]+)", re.IGNORECASE), handle_weather),
    (re.compile(r"погода ([а-яА-Яa-zA-Z\-\s]+)", re.IGNORECASE), handle_weather),
    (re.compile(r"(время|который час|сколько времени)", re.IGNORECASE), handle_time),
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), handle_addition),
    (re.compile(r"(\d+)\s*\-\s*(\d+)"), handle_subtraction),
    (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), handle_set_name),
]

bot_instance = ChatBot()

def process_message(message):
    message_clean = message.strip()
    
    bot_response = bot_instance.process(message_clean)
    if bot_response:
        return bot_response
    
    for pattern, handler in patterns:
        match = pattern.search(message_clean)
        if match:
            import inspect
            sig = inspect.signature(handler)
            if len(sig.parameters) > 0:
                return handler(match)
            else:
                return handler()
    
    return "Я не понимаю запрос."

init_db()