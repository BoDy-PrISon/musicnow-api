import os

import librosa
import numpy as np
import tensorflow as tf
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../Models/genre_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "../Models/genre_encoder.pkl")

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_genre(audio_path: str) -> dict:
    model = load_model()

    # Загрузка энкодера
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    genres = encoder.classes_

    # Загрузка и обработка аудио
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    MAX_SAMPLES = 22050 * 30
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    elif len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))

    # Генерация Mel-спектрограммы
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Приведение к размеру 128x128
    target_length = 128
    if mel_db.shape[1] < target_length:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_length - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :target_length]

    # Подготовка входных данных для модели
    input_tensor = np.expand_dims(mel_db, axis=(0, -1))

    # Предсказание
    pred = model.predict(input_tensor)[0]
    top_index = np.argmax(pred)
    genre = genres[top_index]

    return {
        "genre": genre,
        "confidence": float(pred[top_index])  # опционально, если нужно
    }



# Пример использования
# print(predict_genre(audio_path="../000002.mp3"))