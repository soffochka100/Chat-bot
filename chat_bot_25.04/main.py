import sys
from bot_core import ChatBot, process_message, log_message, init_db, load_bert, init_tts
from voice_asr import load_whisper_model, listen

def main():
    print("=" * 70)
    print("ЧАТ-БОТ НА BERT + TTS + ASR")
    print("=" * 70)
    
    print("Инициализация базы данных...")
    init_db()
    
    print("Загрузка BERT модели...")
    bert_loaded = load_bert()
    
    if not bert_loaded:
        print("\n BERT модель не найдена!")
        print(" Для обучения модели выполните:")
        print("   python train_bert.py")
        return
    
    print(" BERT модель успешно загружена!")
    
    print("\n Инициализация Whisper ASR...")
    try:
        load_whisper_model("base")
        voice_available = True
        print(" Whisper ASR готов!")
    except Exception as e:
        print(f" Ошибка загрузки Whisper: {e}")
        voice_available = False
    
    print("\n Инициализация TTS...")
    tts_loaded = init_tts()
    if tts_loaded:
        print(" TTS готов!")
    else:
        print(" TTS не инициализирован.")
    
    user_id = 123456789
    bot = ChatBot(user_id=user_id)
    
    print("\n" + "=" * 70)
    greeting = bot.greet()
    print(" Бот:", greeting)
    if tts_loaded:
        bot.speak_response(greeting)
    print("=" * 70)
    
    print("\n КОМАНДЫ:")
    print("   • 'voice' или 'голос' - включить голосовой ввод (одно сообщение)")
    print("   • 'voiceon' или 'голосвкл' - включить голосовой режим (все сообщения)")
    print("   • 'text' или 'текст' - вернуться в текстовый режим")
    print("   • 'выход', 'exit', 'quit' - завершить работу")
    print("   • 'отмена', 'стоп' - прервать текущий диалог")
    print("")
    
    global_mode = "text"
    temp_voice_mode = False
    
    print(f"🎯 Текущий режим: ТЕКСТОВЫЙ ввод")
    print("   (напишите 'voice' для голосового ввода одного сообщения)")
    print("-" * 70)
    
    while True:
        try:
            if global_mode == "voice":
                print("\n [Голосовой режим] Ожидание...")
                
                print("Нажмите ENTER чтобы начать говорить (или напишите 'text' для выхода из голосового режима)...")
                confirm = input().strip().lower()
                
                if confirm == 'text' or confirm == 'текст':
                    global_mode = "text"
                    temp_voice_mode = False
                    print("Переключено в ТЕКСТОВЫЙ режим")
                    continue
                
                print("Начинайте говорить...")
                user_input = listen(seconds=5, dynamic=True)
                
                if not user_input:
                    print(" Не удалось распознать речь. Попробуйте ещё раз или напишите 'text' для выхода.")
                    continue
                    
            elif temp_voice_mode:
                print("\n [Голосовое сообщение]")
                
                input("Нажмите ENTER чтобы начать говорить...")
                
                print("Начинайте говорить...")
                user_input = listen(seconds=5, dynamic=True)
                
                temp_voice_mode = False
                print("Возврат в ТЕКСТОВЫЙ режим")
                
                if not user_input:
                    print("Не удалось распознать речь. Попробуйте ещё раз текстом.")
                    continue
            else:
                user_input = input(" Вы: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ('выход', 'exit', 'quit'):
                print("Бот: До свидания!")
                if tts_loaded:
                    bot.speak_response("До свидания!")
                break
            
            if user_input.lower() in ('voice', 'голос') and voice_available:
                temp_voice_mode = True
                print("Будет использован голосовой ввод для СЛЕДУЮЩЕГО сообщения")
                print("   (напишите что-нибудь ещё раз для отмены)")
                continue
            
            if user_input.lower() in ('voiceon', 'голосвкл') and voice_available:
                global_mode = "voice"
                temp_voice_mode = False
                print("Включен ПОСТОЯННЫЙ голосовой режим")
                print("   (напишите 'text' для выхода из голосового режима)")
                continue
            
            if user_input.lower() in ('text', 'текст'):
                global_mode = "text"
                temp_voice_mode = False
                print("Переключено в ТЕКСТОВЫЙ режим")
                continue
            
            if user_input.lower() in ('отмена', 'стоп', 'cancel'):
                if bot.state.value != "start":
                    bot.reset_state()
                    response = "Диалог прерван. Чем могу помочь?"
                else:
                    response = "Нет активного диалога для отмены."
                print("Бот:", response)
                if tts_loaded:
                    bot.speak_response(response)
                continue
            
            response = process_message(user_input, bot)
            print("Бот:", response)
            
            log_message(user_input, response)
                
        except KeyboardInterrupt:
            print("\n Бот: До свидания!")
            if tts_loaded:
                bot.speak_response("До свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            print("Бот: Попробуйте ещё раз")

if __name__ == "__main__":
    main()