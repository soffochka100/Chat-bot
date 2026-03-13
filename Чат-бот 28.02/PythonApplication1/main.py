from bot_core import init_db, process_message, log_message, ChatBot, save_user, get_user_by_name
from weather_api import get_weather

DB_NAME = "bot.db"

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