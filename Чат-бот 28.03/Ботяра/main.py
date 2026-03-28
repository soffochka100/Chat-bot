
import sys
from bot_core import ChatBot, process_message, log_message, init_db, analyze_text_with_spacy, load_ml_model

def main():
    """
    Главная функция для запуска бота в консоли
    """
    print("Инициализация базы данных...")
    init_db()
    
    print("Загрузка ML-модели...")
    model_loaded = load_ml_model()
    
    if model_loaded:
        print("ML-модель успешно загружена и готова к работе")
    else:
        print("ВНИМАНИЕ: ML-модель не найдена. Бот будет работать в резервном режиме с регулярными выражениями.")
        print("Для обучения модели выполните: python train_model.py")
    
    user_id = 123456789 
    
    bot = ChatBot(user_id=user_id)
    
    print("=" * 70)
    print(" Бот: Здравствуйте!")
    print("=" * 70)
    print(" Подсказки:")
    print(" - 'выход', 'exit', 'quit' - завершить работу")
    print(" - 'отмена', 'стоп' - прервать текущий диалог")
    print(" - 'пока', 'до свидания' - попрощаться")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
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
                print(" Бот: Всего доброго! Заходите ещё.")
                break
            
            response = process_message(user_input, bot)
            print(" Бот:", response)
            
            log_message(user_input, response)
                
        except KeyboardInterrupt:
            print("\n Бот: Работа завершена. До свидания!")
            break
        except Exception as e:
            print(f" Бот: Произошла ошибка: {e}")
            print(" Бот: Попробуйте ещё раз")

if __name__ == "__main__":
    main()