import re
import string
from datetime import datetime


def handle_greeting():
    return "Здравствуйте! Чем могу помочь?"


def handle_farewell():
    return "До свидания!"


def handle_weather(match):
    city = match.group(1)
    return f"Погода в городе {city}: солнечно (демо-режим)."


def handle_time(match):
    current_time = datetime.now().strftime("%H:%M")
    return f"Текущее время: {current_time}"


def handle_addition(match):
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a + b}"


def handle_subtraction(match):
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a - b}"


def log_message(user, bot):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] USER: {user}\n")
        f.write(f"[{datetime.now()}] BOT: {bot}\n")


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
        return f"Приятно познакомиться, {self.name}!"

    def process(self, message):
        message = message.lower().strip()
        message = message.translate(str.maketrans('', '', string.punctuation))
        
        for pattern, handler in self.patterns:
            match = pattern.search(message)
            if match:
                return handler(match)
        return "Не понимаю запрос."


patterns = [
    (re.compile(r"^(привет|здравствуй|добрый день)$", re.IGNORECASE), handle_greeting),
    (re.compile(r"^(пока|до свидания)$", re.IGNORECASE), handle_farewell),
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), handle_weather),
    (re.compile(r"время|который час|сколько времени", re.IGNORECASE), handle_time),
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), handle_addition),
    (re.compile(r"(\d+)\s*\-\s*(\d+)"), handle_subtraction),
    (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), lambda match: f"Приятно познакомиться, {match.group(1)}!"),
]


def process_message(message: str):
    message = message.strip()

    for pattern, handler in patterns:
        match = pattern.search(message)
        if match:
            if callable(handler) and handler.__code__.co_argcount > 0:
                return handler(match)
            return handler()

    return "Я не понимаю запрос."


if __name__ == "__main__":
    while True:
        user_input = input("Вы: ")
        response = process_message(user_input)
        print("Бот:", response)
        log_message(user_input, response)