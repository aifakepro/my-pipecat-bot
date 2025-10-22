from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import google.generativeai as genai
import requests
import tempfile
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Настройте API ключи
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

# Инициализация
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/')
def home():
    return jsonify({"status": "Voice Assistant API running"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Транскрибация аудио через OpenAI Whisper"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        # Транскрибация через OpenAI Whisper
        with open(temp_audio_path, 'rb') as audio:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="ru"  # Украинский/Русский
            )
        
        # Удаляем временный файл
        os.unlink(temp_audio_path)
        
        return jsonify({"text": transcript.text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    """Обработка текста через Google Gemini"""
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        # Инициализация модели Gemini
        model = genai.GenerativeModel('gemini-pro')
        
        # Генерация ответа
        response = model.generate_content(user_text)
        
        return jsonify({"response": response.text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speak', methods=['POST'])
def text_to_speech():
    """Озвучка текста через ElevenLabs"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # ElevenLabs API
        VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice (можете изменить)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            # Возвращаем аудио файл
            return send_file(
                BytesIO(response.content),
                mimetype='audio/mpeg',
                as_attachment=False
            )
        else:
            return jsonify({"error": "ElevenLabs API error"}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
