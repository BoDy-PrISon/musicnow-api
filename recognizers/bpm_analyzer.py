import tempfile
import librosa
import soundfile as sf
import os
import logging

logger = logging.getLogger(__name__)

def estimate_bpm(audio_path: str) -> dict:
    try:
        # Принудительно конвертируем в WAV если нужно
        if not audio_path.lower().endswith('.wav'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                logger.info(f"🔄 Конвертируем в WAV: {audio_path} -> {tmp_wav.name}")
                y, sr = librosa.load(audio_path, sr=None)
                sf.write(tmp_wav.name, y, sr)
                audio_path = tmp_wav.name

        logger.info(f"📀 Загружаем аудиофайл: {audio_path}")
        y, sr = librosa.load(audio_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo), 2)
        logger.info(f"🎶 BPM: {bpm}")
        return {"bpm": bpm}

    except Exception as e:
        logger.exception(f"BPM Error: {str(e)}")
        return {"bpm": 0.0}



