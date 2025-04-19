import requests


# Функция для распознавания трека через AudD + анализ настроения через Last.fm
def recognize_song_with_mood(file_path: str, audd_token: str, lastfm_token: str) -> dict:
    # 1. Распознаем трек через AudD
    audd_url = "https://api.audd.io/"
    with open(file_path, 'rb') as f:
        audd_response = requests.post(
            audd_url,
            files={'file': f},
            data={'api_token': audd_token, 'return': 'spotify'}
        ).json()

    if not audd_response.get('result'):
        return {'status': 'error', 'message': 'Трек не распознан'}

    # 2. Получаем данные о треке
    track_info = {
        'title': audd_response['result']['title'],
        'artist': audd_response['result']['artist'],
        'spotify_id': audd_response['result']['spotify']['id']
    }

    # 3. Определяем настроение через Last.fm
    lastfm_url = "http://ws.audioscrobbler.com/2.0/"
    lastfm_params = {
        'method': 'track.gettoptags',
        'artist': track_info['artist'],
        'track': track_info['title'],
        'api_key': lastfm_token,
        'format': 'json'
    }

    lastfm_response = requests.get(lastfm_url, params=lastfm_params).json()
    mood = analyze_mood_from_tags(lastfm_response)

    # 4. Добавляем настроение в результат
    track_info.update({
        'status': 'success',
        'mood': mood,
        'lastfm_tags': [tag['name'] for tag in lastfm_response.get('toptags', {}).get('tag', [])]
    })

    return track_info


# Вспомогательная функция для анализа тегов
def analyze_mood_from_tags(lastfm_data):
    mood_mapping = {
        'sad': ['sad', 'depressing', 'melancholic'],
        'happy': ['happy', 'joyful', 'uplifting'],
        'energetic': ['energetic', 'party', 'dance'],
        'calm': ['calm', 'relaxing', 'peaceful']
    }

    tags = [tag['name'].lower() for tag in lastfm_data.get('toptags', {}).get('tag', [])]

    for mood, keywords in mood_mapping.items():
        if any(keyword in tags for keyword in keywords):
            return mood

    return 'neutral'


