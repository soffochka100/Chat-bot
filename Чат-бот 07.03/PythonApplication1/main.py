import sys
from bot_core import ChatBot, process_message, log_message, init_db, analyze_text_with_spacy

def main():
    print("Инициализация базы данных...")
    init_db()

    bot = ChatBot(user_id=123456789)
    
    print("=" * 70)
    print(" Бот: Здравствуйте!")
    print(" Вы можете:")
    print("  - Спросить о погоде (например: 'Какая погода в Москве?')")
    print("  - Сложить числа (например: '5 + 3')")
    print("  - Представиться (например: 'Меня зовут Александр')")
    print("=" * 70)
    print("Для выхода напишите 'пока' или 'до свидания'")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if user_input.lower() in ('выход', 'exit', 'quit'):
                print("Бот: До свидания!")
                break
            
            response = process_message(user_input, bot)
            print(" Бот:", response)
            
            log_message(user_input, response)
            
            if any(word in user_input.lower() for word in ['пока', 'до свидания']):
                print(" Бот: Всего доброго! Заходите ещё.")
                break
                
        except KeyboardInterrupt:
            print("\n Бот: Работа завершена. До свидания!")
            break
        except Exception as e:
            print(f" Бот: Произошла ошибка: {e}")
            print(" Бот: Попробуйте ещё раз или напишите 'помощь'")

if __name__ == "__main__":
    main()