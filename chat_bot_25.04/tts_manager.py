import pyttsx3
import threading
import re

class TTSManager:
    def __init__(self):
        print("Инициализация TTS (pyttsx3)...")
        self.lock = threading.Lock()
        print("TTS готов!")
    
    def normalize_text(self, text: str) -> str:
        """Очистка текста от эмодзи и лишних символов"""
        clean_text = re.sub(r'[^\w\s\.\,\!\?\-\—]', '', text)
        return clean_text.strip()
    
    def speak(self, text: str, async_mode: bool = True):
        """Озвучивание текста"""
        if not text:
            return
        
        clean_text = self.normalize_text(text)
        if not clean_text:
            return
        
        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(clean_text,))
            thread.daemon = True
            thread.start()
        else:
            self._speak_sync(clean_text)
    
    def _speak_sync(self, text: str):
        """Синхронное озвучивание - создаём новый движок для каждого вызова"""
        with self.lock:
            engine = None
            try:
                engine = pyttsx3.init()
                
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'russian' in voice.name.lower() or 'русский' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                
                engine.setProperty('rate', 270)  # Скорость речи
                engine.setProperty('volume', 0.9)  # Громкость
                
                engine.say(text)
                engine.runAndWait()
                
            except Exception as e:
                print(f"Ошибка воспроизведения: {e}")
            finally:
                if engine:
                    try:
                        engine.stop()
                    except:
                        pass