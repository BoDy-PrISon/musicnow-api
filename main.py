from fastapi import FastAPI, UploadFile, File
from recognizers.AudD import recognize_song_with_mood
from recognizers.bpm_analyzer import estimate_bpm
from recognizers.genre_classifier import predict_genre
#from recognizers.mood import predict_mood
from recognizers.instrument_detector import detect_instruments

import os

app = FastAPI()

# Папка для загруженных файлов
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Твой API ключ AudD
AUDD_API_TOKEN = "61e9068ba431fc91c9596b86aebc5f5d"
Last_fm_token = "c1107c7c0f1cf4f9b31f7d4b84a98997"

@app.post("/recognize")
async def recognize_song(file: UploadFile = File(...)):
    # Сохраняем файл
    temp_audio = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_audio, "wb") as f:
        f.write(await file.read())

    # Запускаем анализаторы
    results = {
        "metadata": {
            "instruments": detect_instruments(temp_audio),
            #"mood": predict_mood(temp_audio),
            "bpm": estimate_bpm(temp_audio),
            "genre": predict_genre(temp_audio),
        },
        "audd": recognize_song_with_mood(temp_audio, AUDD_API_TOKEN, Last_fm_token),
    }

    # Удаляем временный файл
    os.remove(temp_audio)
    return results