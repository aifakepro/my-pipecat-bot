from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import requests
import os
from io import BytesIO
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Настройка API ключей из окружения
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/status')
def status():
    return jsonify({"status": "Voice Assistant API running"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    try:
        audio_file.seek(0)
    except Exception:
        pass

    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    content_type = audio_file.content_type or ""
    if content_type.lower().startswith("audio/webm") and "codecs" not in content_type.lower():
        content_type = "audio/webm; codecs=opus"
    if not content_type:
        content_type = "audio/webm; codecs=opus"

    url = "https://api.deepgram.com/v1/listen?model=nova-2-general&language=ru"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}" if DEEPGRAM_API_KEY else "",
        "Content-Type": content_type
    }

    try:
        resp = requests.post(url, headers=headers, data=audio_bytes, timeout=30)
    except requests.RequestException as e:
        return jsonify({"error": "Deepgram request failed", "details": str(e)}), 502

    if resp.status_code != 200:
        return jsonify({
            "error": "Deepgram API error",
            "status": resp.status_code,
            "details": resp.text
        }), 500

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

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        app.logger.warning("Нет текста в запросе /chat")
        return jsonify({"error": "No text provided"}), 400

    user_text = data.get('text', '')
    app.logger.info(f"Запрос к Gemini: {user_text}")

    if not GEMINI_API_KEY:
        app.logger.error("Gemini API key не настроен")
        return jsonify({"error": "Gemini API key not configured"}), 500

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')

        response = model.generate_content(user_text)
        app.logger.info(f"Ответ Gemini (raw): {response}")

        resp_text = getattr(response, "text", None)
        if not resp_text:
            resp_text = str(response)
            app.logger.warning(f"Gemini вернул объект без .text, fallback к str: {resp_text}")

        app.logger.info(f"Ответ Gemini (текст): {resp_text}")
        return jsonify({"response": resp_text})

    except Exception as e:
        app.logger.exception("Ошибка при вызове Gemini")
        return jsonify({"error": "Gemini API error", "details": str(e)}), 500

@app.route('/speak', methods=['POST'])
def text_to_speech():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        app.logger.error("Нет текста в запросе /speak")
        return jsonify({"error": "No text provided"}), 400

    text = data.get('text', '').strip()
    
    # Проверка на пустой текст после очистки
    if not text:
        app.logger.error("Текст пустой после strip()")
        return jsonify({"error": "Empty text provided"}), 400
    
    app.logger.info(f"Запрос на озвучку: {text[:100]}...")  # Логируем первые 100 символов

    if not ELEVENLABS_API_KEY:
        app.logger.error("ElevenLabs API key не настроен")
        return jsonify({"error": "ElevenLabs API key not configured"}), 500

    # Используем стандартный голос Rachel (англ) или Adam (англ)
    # Для русского языка лучше использовать голос, поддерживающий multilingual
    VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - поддерживает multilingual
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    # Ограничиваем длину текста (ElevenLabs имеет лимиты)
    max_chars = 5000
    if len(text) > max_chars:
        text = text[:max_chars]
        app.logger.warning(f"Текст обрезан до {max_chars} символов")
    
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    app.logger.info(f"Отправка запроса в ElevenLabs с voice_id={VOICE_ID}")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)  # Увеличен таймаут
        
        app.logger.info(f"ElevenLabs status: {resp.status_code}")
        
        if resp.status_code != 200:
            error_text = resp.text
            app.logger.error(f"ElevenLabs error: {error_text}")
            
            # Пытаемся распарсить JSON ошибку
            try:
                error_json = resp.json()
                app.logger.error(f"ElevenLabs error JSON: {error_json}")
            except:
                pass
            
            return jsonify({
                "error": "ElevenLabs API error",
                "status": resp.status_code,
                "details": error_text
            }), 500

        audio_bytes = resp.content
        
        if not audio_bytes:
            app.logger.error("ElevenLabs вернул пустой audio")
            return jsonify({"error": "Empty audio response from ElevenLabs"}), 500
        
        app.logger.info(f"Успешно получено аудио, размер: {len(audio_bytes)} байт")
        return send_file(BytesIO(audio_bytes), mimetype="audio/mpeg")
        
    except requests.RequestException as e:
        app.logger.exception("Ошибка запроса к ElevenLabs")
        return jsonify({"error": "ElevenLabs request failed", "details": str(e)}), 502
    except Exception as e:
        app.logger.exception("Неожиданная ошибка в /speak")
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Для production используем debug=False
    app.run(host='0.0.0.0', port=port, debug=False)
