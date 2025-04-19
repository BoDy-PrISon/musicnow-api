import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# === Параметры ===
DATA_DIR = "D:/fma_large"
META_PATH = "fma_metadata"
OUTPUT_MODEL = "genre_model.h5"
OUTPUT_ENCODER = "genre_encoder.pkl"
SR = 22050
DURATION = 30
N_MELS = 128
IMG_SIZE = 128
MAX_TRACKS = 15000 # Установи None, чтобы использовать все треки. Пример: 5000 для тестов.

# === Загрузка метаданных ===
print("Загрузка метаданных...")
tracks = pd.read_csv(os.path.join(META_PATH, "tracks.csv"), index_col=0, header=[0, 1])
tracks = tracks.loc[tracks.index < 106574]
tracks = tracks[tracks['set', 'subset'] == 'large']
tracks = tracks[tracks['track', 'genre_top'].notnull()]
track_genres = tracks['track', 'genre_top']

if MAX_TRACKS is not None:
    track_genres = track_genres.sample(n=MAX_TRACKS, random_state=42)

# === Функция извлечения спектрограммы ===
def extract_mel(file_path):
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    y = librosa.util.fix_length(y, size=SR * DURATION)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < IMG_SIZE:
        mel_db = np.pad(mel_db, ((0, 0), (0, IMG_SIZE - mel_db.shape[1])))
    return mel_db[:, :IMG_SIZE]

# === Подготовка данных ===
print("Генерация спектрограмм...")
X = []
y = []

for track_id in track_genres.index:
    genre = track_genres.loc[track_id]
    folder = f"{track_id:06d}"[:3]
    file_path = os.path.join(DATA_DIR, folder, f"{track_id:06d}.mp3")
    if not os.path.exists(file_path):
        continue
    try:
        mel = extract_mel(file_path)
        if mel.shape == (N_MELS, IMG_SIZE):
            X.append(mel)
            y.append(genre)
    except Exception as e:
        print(f"[Ошибка] {track_id}: {e}")

# === Фильтрация жанров с < 2 примерами ===
X = np.array(X)
y = np.array(y)
genre_counts = Counter(y)
valid_genres = [genre for genre, count in genre_counts.items() if count >= 2]
mask = np.isin(y, valid_genres)
X = X[mask]
y = y[mask]

# === Энкодинг ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
X = X[..., np.newaxis]  # (samples, 128, 128, 1)

print(f"Данных после фильтрации: {X.shape}, Жанров: {len(encoder.classes_)}")

# === Разделение ===
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoded)

# === Модель ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Обучение ===
print("Обучение модели...")
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

# === Сохранение ===
print("Сохранение модели и энкодера...")
model.save(OUTPUT_MODEL)
with open(OUTPUT_ENCODER, "wb") as f:
    pickle.dump(encoder, f)

print("Готово!")
