
import sys
from bot_core import ChatBot, process_message, log_message, init_db, load_bert, is_bert_ready

def main():
    print("=" * 70)
    print("ЧАТ-БОТ НА BERT")
    print("=" * 70)
    
    print("Инициализация базы данных...")
    init_db()
    
    print("Загрузка BERT модели...")
    bert_loaded = load_bert()
    
    if not bert_loaded:
        print("\n BERT модель не найдена!")
        print(" Для обучения модели выполните:")
        print("   python train_bert.py")
        print("\nПосле обучения запустите бота снова.")
        return
    
    print("BERT модель успешно загружена!")
    
    user_id = 123456789 
    bot = ChatBot(user_id=user_id)
    
    print("\n" + "=" * 70)
    print(" Бот: Здравствуйте! Я чат-бот.")
    print("=" * 70)
    print(" Команды:")
    print("   - 'выход', 'exit', 'quit' - завершить работу")
    print("   - 'отмена', 'стоп' - прервать текущий диалог")
    print("")
    print(" Что я умею:")
    print("    Рассказывать о погоде")
    print("    Складывать числа")
    print("    Запоминать ваше имя")
    print("    Отвечать на приветствия и прощания")
    print("=" * 70)
    print()
    
    while True:
        try:
            user_input = input(" Вы: ").strip()
            
            if user_input.lower() in ('выход', 'exit', 'quit'):
                print(" Бот: До свидания!")
                break
                
            if user_input.lower() in ('отмена', 'стоп', 'cancel'):
                if bot.state.value != "start":
                    bot.reset_state()
                    print(" Бот: Диалог прерван. Чем могу помочь?")
                else:
                    print(" Бот: Нет активного диалога для отмены.")
                continue
            
            if user_input.lower() in ('пока', 'до свидания', 'всего хорошего'):
                response = process_message(user_input, bot)
                print(" Бот:", response)
                break
            
            response = process_message(user_input, bot)
            print(" Бот:", response)
            
            log_message(user_input, response)
                
        except KeyboardInterrupt:
            print("\n Бот: Работа завершена. Всего хорошего! 👋")
            break
        except Exception as e:
            print(f" Ошибка: {e}")
            print(" Бот: Попробуйте ещё раз")

if __name__ == "__main__":
    main()