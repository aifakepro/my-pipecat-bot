from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import requests
import os
from io import BytesIO
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Настройка API ключей из окружения (как у тебя было)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

@app.route('/')
def home():
    # сохраняем поведение: отдаём index.html из текущей директории
    return send_from_directory('.', 'index.html')

@app.route('/status')
def status():
    return jsonify({"status": "Voice Assistant API running"})

# Транскрибация аудио через Deepgram — интерфейс не менялся: POST /transcribe -> {"text": "..."}
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    # убедимся, что поток в начале
    try:
        audio_file.seek(0)
    except Exception:
        pass

    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    # определяем content-type; для webm добавляем hint codecs=opus, если его нет
    content_type = audio_file.content_type or ""
    if content_type.lower().startswith("audio/webm") and "codecs" not in content_type.lower():
        content_type = "audio/webm; codecs=opus"
    if not content_type:
        # fallback на WebM/Opus — чаще всего браузер даёт такой поток
        content_type = "audio/webm; codecs=opus"

    # явная модель + язык для стабильности; адрес соответствует документации
    url = "https://api.deepgram.com/v1/listen?model=nova-2-general&language=ru"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}" if DEEPGRAM_API_KEY else "",
        "Content-Type": content_type
    }

    try:
        resp = requests.post(url, headers=headers, data=audio_bytes, timeout=30)
    except requests.RequestException as e:
        return jsonify({"error": "Deepgram request failed", "details": str(e)}), 502

    # если Deepgram вернул ошибку — отдаём её текст, чтобы фронт/логи видели причину
    if resp.status_code != 200:
        return jsonify({
            "error": "Deepgram API error",
            "status": resp.status_code,
            "details": resp.text
        }), 500

    # парсим результат как раньше — возвращаем {"text": "..."}
    try:
        data = resp.json()
    except ValueError:
        return jsonify({"error": "Deepgram returned non-JSON response", "raw": resp.text}), 500

    transcript = (
        data.get('results', {})
            .get('channels', [{}])[0]
            .get('alternatives', [{}])[0]
            .get('transcript', '')
    )

    if not transcript:
        return jsonify({"error": "Empty transcript", "raw": data}), 500

    return jsonify({"text": transcript})

# Чат с Gemini — интерфейс не менялся: POST /chat {text: "..."} -> {"response": "..."}
@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        app.logger.warning("Нет текста в запросе /chat")
        return jsonify({"error": "No text provided"}), 400

    user_text = data.get('text', '')
    app.logger.info(f"Запрос к Gemini: {user_text}")  # Логируем текст запроса

    if not GEMINI_API_KEY:
        app.logger.error("Gemini API key не настроен")
        return jsonify({"error": "Gemini API key not configured"}), 500

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')

        response = model.generate_content(user_text)
        app.logger.info(f"Ответ Gemini (raw): {response}")  # Логируем полный объект

        resp_text = getattr(response, "text", None)
        if not resp_text:
            resp_text = str(response)  # fallback, если .text нет
            app.logger.warning(f"Gemini вернул объект без .text, fallback к str: {resp_text}")

        app.logger.info(f"Ответ Gemini (текст): {resp_text}")  # Логируем текст для фронта
        return jsonify({"response": resp_text})

    except Exception as e:
        app.logger.exception("Ошибка при вызове Gemini")  # Полный трейс в логах
        return jsonify({"error": "Gemini API error", "details": str(e)}), 500


# Озвучка текста через ElevenLabs — интерфейс не менялся: POST /speak {text: "..."} -> audio/mpeg
@app.route('/speak', methods=['POST'])
def text_to_speech():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data.get('text', '')
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "ElevenLabs API key not configured"}), 500

    VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as e:
        return jsonify({"error": "ElevenLabs request failed", "details": str(e)}), 502

    if resp.status_code != 200:
        return jsonify({
            "error": "ElevenLabs API error",
            "status": resp.status_code,
            "details": resp.text
        }), 500

    # возвращаем байты mp3 на фронт так же, как раньше (audio/mpeg)
    audio_bytes = resp.content
    return send_file(BytesIO(audio_bytes), mimetype="audio/mpeg")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
