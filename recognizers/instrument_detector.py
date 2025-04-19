import tensorflow_hub as hub
import soundfile as sf
import numpy as np

# Загрузка YAMNet
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_names = [line.strip() for line in open(model.class_map_path().numpy())]


def detect_instruments(audio_path: str, top_n=5) -> dict:
    waveform, sr = sf.read(audio_path)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)  # Конвертируем в моно

    # Предсказание
    scores, _, _ = model(waveform)
    mean_scores = np.mean(scores, axis=0)
    top_indices = np.argsort(mean_scores)[-top_n:][::-1]
    top_classes = [class_names[i] for i in top_indices]

    return {
        "instruments": top_classes
    }
