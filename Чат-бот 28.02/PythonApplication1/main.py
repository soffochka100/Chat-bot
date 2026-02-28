import sqlite3
from bot_core import process_message, log_message, ChatBot
from weather_api import get_weather

DB_NAME = "bot.db"


def init_db():
    """Инициализация базы данных"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(" База данных успешно инициализирована")

def save_user(user_id, name):
    """Сохранение пользователя в БД"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("INSERT OR REPLACE INTO users (user_id, name) VALUES (?, ?)",
                   (user_id, name))

    conn.commit()
    conn.close()
    print(f" Пользователь {name} (ID: {user_id}) сохранён в БД")

def get_user(user_id):
    """Получение имени пользователя из БД по ID"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None

def run_console_bot():
    """Запуск консольной версии бота"""
    print("=" * 50)
    print(" Чат-бот запущен. Команды:")
    print("  • привет / здравствуй")
    print("  • пока / до свидания")
    print("  • погода в [город]")
    print("  • время / который час")
    print("  • 5 + 3 (сложение)")
    print("  • 10 - 4 (вычитание)")
    print("  • меня зовут [имя]")
    print("=" * 50)
    
    class_bot = ChatBot()
    
    while True:
        try:
            user_input = input("\n Вы: ")
            
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print(" Бот: До свидания!")
                break
            
            response = process_message(user_input)
            
            print(f" Бот: {response}")
            
            log_message(user_input, response)
            
        except KeyboardInterrupt:
            print("\n Бот: До свидания!")
            break
        except Exception as e:
            print(f" Ошибка: {e}")

if __name__ == "__main__":
    init_db()
    run_console_bot()