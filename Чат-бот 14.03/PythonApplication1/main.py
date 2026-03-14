import sys
from bot_core import ChatBot, process_message, log_message, init_db, DialogManager

def main():
    print("Инициализация базы данных...")
    init_db()

    user_id = 123456789
    bot = ChatBot(user_id=user_id)
    
    print("=" * 70)
    print(" Бот: Здравствуйте!")
    print(" Вы можете:")
    print("  - Спросить о погоде (например: 'Какая погода?')")
    print("  - Уточнить город (бот спросит сам)")
    print("  - Уточнить дату (бот спросит сам)")
    print("  - Сложить числа (например: '10+5')")
    print("  - Представиться (например: 'Меня зовут Даша')")
    print("=" * 70)
    print(" Для выхода напишите 'пока' или 'до свидания'")
    print(" Для отмены диалога напишите 'отмена'")
    print("=" * 70)
    
    while True:
        try:
            user_input = input(" Вы: ").strip()
            
            if user_input.lower() in ('выход', 'exit', 'quit'):
                print(" Бот: До свидания!")
                break
            
            response = process_message(user_input, bot)
            print(" Бот:", response)
            
            log_message(user_input, response)
            
            if any(word in user_input.lower() for word in ['пока', 'до свидания']):
                print(" Бот: Всего доброго!")
                break
                
        except KeyboardInterrupt:
            print("\n Бот: Работа завершена. До свидания!")
            break
        except Exception as e:
            print(f" Бот: Произошла ошибка: {e}")
            print(" Бот: Попробуйте ещё раз или напишите 'помощь'")

if __name__ == "__main__":
    main()