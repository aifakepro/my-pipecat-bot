from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import tempfile
import os
from io import BytesIO
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# === API Keys ===
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# === Проверка наличия ключей ===
missing = [k for k, v in {
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY
}.items() if not v]
if missing:
    print(f"[WARN] Missing API keys: {', '.join(missing)}")


@app.route('/')
def home():
    return jsonify({"message": "Voice Assistant API is running"})


@app.route('/status')
def status():
    return jsonify({"status": "ok"})


# === ТРАНСКРИПЦИЯ через Deepgram ===
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_file.seek(0)
    audio_bytes = audio_file.read()

    # Определяем тип контента
    content_type = audio_file.content_type or "audio/webm; codecs=opus"

    url = "https://api.deepgram.com/v1/listen?model=nova-2-general&language=ru"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type
    }

    try:
        resp = requests.post(url, headers=headers, data=audio_bytes, timeout=30)
    except Exception as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

    if resp.status_code != 200:
        return jsonify({
            "error": "Deepgram API error",
            "status": resp.status_code,
            "details": resp.text
        }), resp.status_code

    try:
        data = resp.json()
        transcript = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        if not transcript:
            return jsonify({"error": "Empty transcript", "raw": data}), 500
        return jsonify({"text": transcript})
    except Exception as e:
        return jsonify({"error": f"JSON parse failed: {str(e)}"}), 500


# === ЧАТ через Gemini ===
@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    user_text = data['text']
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(user_text)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": f"Gemini API failed: {str(e)}"}), 500


# === ОЗВУЧКА через ElevenLabs ===
@app.route('/speak', methods=['POST'])
def text_to_speech():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

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
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

    if resp.status_code != 200:
        return jsonify({
            "error": "ElevenLabs API error",
            "status": resp.status_code,
            "details": resp.text
        }), resp.status_code

    # Сохраняем временный mp3-файл для отдачи клиенту
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp.write(resp.content)
    temp.flush()
    return send_file(temp.name, mimetype="audio/mpeg", as_attachment=False)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
