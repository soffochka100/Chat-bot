import whisper
import sounddevice as sd
import numpy as np
import re

whisper_model = None

def load_whisper_model(model_size="base"):
    global whisper_model
    if whisper_model is None:
        print(f"Загрузка Whisper модели ({model_size})...")
        whisper_model = whisper.load_model(model_size)
        print("Whisper модель загружена!")
    return whisper_model

def record_audio_dynamic(max_seconds=10, silence_threshold=0.01, silence_duration=1.5, fs=16000):
    """
    Запись с автоматическим определением окончания речи
    """
    chunk_size = int(fs * 0.1)
    audio_buffer = []
    silent_chunks = 0
    silence_chunks_needed = int(silence_duration / 0.1)
    has_sound = False
    
    stream = sd.InputStream(samplerate=fs, channels=1, dtype=np.float32)
    stream.start()
    
    total_chunks = 0
    max_chunks = int(max_seconds / 0.1)
    
    print("   ", end="", flush=True)
    
    while total_chunks < max_chunks:
        data, _ = stream.read(chunk_size)
        audio_buffer.append(data.flatten())
        chunk_energy = np.sqrt(np.mean(data**2))
        
        if chunk_energy > silence_threshold:
            if not has_sound:
                print("Запись...", end="", flush=True)
            has_sound = True
            silent_chunks = 0
            print(".", end="", flush=True)
        elif has_sound:
            silent_chunks += 1
            if silent_chunks >= silence_chunks_needed:
                print("\n   Запись остановлена")
                break
        
        total_chunks += 1
    
    stream.stop()
    stream.close()
    
    if not has_sound:
        print("\n   Речь не обнаружена")
        return np.array([]), fs
    
    audio = np.concatenate(audio_buffer)
    print(f"   Длительность: {len(audio)/fs:.1f} сек")
    
    return audio, fs

def clean_asr_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яё0-9\s\.\,\?\!\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def speech_to_text_from_memory(audio, fs=16000, language="ru"):
    global whisper_model
    if whisper_model is None:
        load_whisper_model()
    
    if len(audio) < fs * 0.5:
        return ""
    
    result = whisper_model.transcribe(audio, language=language, fp16=False)
    raw_text = result["text"]
    cleaned_text = clean_asr_text(raw_text)
    return cleaned_text

def listen(seconds=5, dynamic=True):
    """
    Основная функция: запись + распознавание
    """
    if dynamic:
        audio, fs = record_audio_dynamic(max_seconds=seconds)
    else:
        audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        audio = audio.flatten()
        fs = 16000
        print("Запись завершена!")
    
    if len(audio) == 0:
        return ""
    
    text = speech_to_text_from_memory(audio, fs)
    if text:
        print(f"Распознано: {text}")
    else:
        print("Ничего не распознано")
    return text