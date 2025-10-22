from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import google.generativeai as genai
import requests
import tempfile
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Настройка API ключей
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Главная страница
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# Статус API
@app.route('/status')
def status():
    return jsonify({"status": "Voice Assistant API running"})

# Транскрибация аудио через OpenAI Whisper
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
        audio_file.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    with open(temp_audio_path, 'rb') as audio:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            language="ru"
        )

    os.unlink(temp_audio_path)
    return jsonify({"text": transcript.text})

# Чат с Gemini
@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    data = request.get_json()
    user_text = data.get('text', '')
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

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
