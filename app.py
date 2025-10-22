from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import tempfile
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Настройка API ключей
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

# Главная страница
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# Статус API
@app.route('/status')
def status():
    return jsonify({"status": "Voice Assistant API running"})

# Транскрибация аудио через Deepgram
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']

    audio_bytes = audio_file.read()
    url = "https://api.deepgram.com/v1/listen?language=ru-RU"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/webm"
    }

    resp = requests.post(url, headers=headers, data=audio_bytes)
    if resp.status_code != 200:
        return jsonify({"error": "Deepgram API error"}), 500

    transcript = resp.json().get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
    return jsonify({"text": transcript})

# Чат с Gemini
@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    import google.generativeai as genai
    data = request.get_json()
    user_text = data.get('text', '')
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(user_text)
    return jsonify({"response": response.text})

# Озвучка текста через ElevenLabs
@app.route('/speak', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

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

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return send_from_directory(
            '.', 
            BytesIO(response.content),
            mimetype='audio/mpeg'
        )
    else:
        return jsonify({"error": "ElevenLabs API error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
