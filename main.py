import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
import logging
from pathlib import Path

from recognizers.AudD import recognize_song_with_mood
from recognizers.bpm_analyzer import estimate_bpm
from recognizers.genre_classifier import predict_genre
from recognizers.instrument_detector import detect_instruments

# Инициализация логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MusicNow API", version="1.0.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Защита ключей через переменные окружения
AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN", "61e9068ba431fc91c9596b86aebc5f5d")
LASTFM_TOKEN = os.getenv("LASTFM_TOKEN", "c1107c7c0f1cf4f9b31f7d4b84a98997")

# Проверка загрузки моделей при старте
@app.on_event("startup")
async def startup_event():
    try:
        from recognizers.genre_classifier import load_genre_model
        load_genre_model()  # Предзагрузка модели
        logger.info("Модели успешно загружены")
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {str(e)}")

@app.post("/recognize", response_model=Dict[str, Any])
async def recognize_song(file: UploadFile = File(...)):
    """Анализ аудиофайла: метаданные, настроение, BPM, жанр"""
    try:
        # Валидация файла
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            raise HTTPException(400, "Поддерживаются только MP3/WAV файлы")

        # Сохранение временного файла
        temp_audio = UPLOAD_DIR / file.filename
        async with aiofiles.open(temp_audio, 'wb') as f:
            await f.write(await file.read())

        # Параллельный анализ
        results = {
            "metadata": await analyze_audio(temp_audio),
            "audd": await recognize_with_audd(temp_audio)
        }

        return results

    except Exception as e:
        logger.error(f"Ошибка обработки файла: {str(e)}")
        raise HTTPException(500, "Internal Server Error")

    finally:
        # Очистка временных файлов
        if temp_audio.exists():
            temp_audio.unlink()

async def analyze_audio(file_path: Path) -> Dict[str, Any]:
    """Анализ аудио: инструменты, BPM, жанр"""
    try:
        return {
            "instruments": detect_instruments(str(file_path)),
            "bpm": estimate_bpm(str(file_path)),
            "genre": predict_genre(str(file_path)),
        }
    except Exception as e:
        logger.error(f"Audio analysis error: {str(e)}")
        return {"error": str(e)}

async def recognize_with_audd(file_path: Path) -> Dict[str, Any]:
    """Распознавание через AudD + анализ настроения"""
    try:
        return recognize_song_with_mood(
            str(file_path),
            AUDD_API_TOKEN,
            LASTFM_TOKEN
        )
    except Exception as e:
        logger.error(f"AudD error: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера"""
    return {
        "status": "OK",
        "models_loaded": True
    }