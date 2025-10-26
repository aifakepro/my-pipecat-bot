from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import requests
import os
from io import BytesIO
import google.generativeai as genai
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# API keys configuration from environment
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/status')
def status():
    return jsonify({"status": "Voice Assistant API running"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribes audio using Deepgram API"""
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

    url = "https://api.deepgram.com/v1/listen?model=nova-2-general&language=en"
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
    """Gets response from Gemini AI"""
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        app.logger.warning("No text in /chat request")
        return jsonify({"error": "No text provided"}), 400

    user_text = data.get('text', '')
    app.logger.info(f"Request to Gemini: {user_text}")

    if not GEMINI_API_KEY:
        app.logger.error("Gemini API key not configured")
        return jsonify({"error": "Gemini API key not configured"}), 500

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')

        response = model.generate_content(user_text)
        app.logger.info(f"Gemini response (raw): {response}")

        resp_text = getattr(response, "text", None)
        if not resp_text:
            resp_text = str(response)
            app.logger.warning(f"Gemini returned object without .text, fallback to str: {resp_text}")

        app.logger.info(f"Gemini response (text): {resp_text}")
        return jsonify({"response": resp_text})

    except Exception as e:
        app.logger.exception("Error calling Gemini")
        return jsonify({"error": "Gemini API error", "details": str(e)}), 500

@app.route('/speak', methods=['POST'])
def text_to_speech():
    """Converts text to audio using gTTS"""
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        app.logger.error("No text in /speak request")
        return jsonify({"error": "No text provided"}), 400

    text = data.get('text', '').strip()
    
    if not text:
        app.logger.error("Text empty after strip()")
        return jsonify({"error": "Empty text provided"}), 400
    
    # Clean text from markdown formatting
    import re
    # Remove ALL asterisks (*, **, ***, etc)
    text = re.sub(r'\*+', '', text)
    # Remove hashes for headers (# Header)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove ALL underscores (_, __, etc)
    text = re.sub(r'_+', '', text)
    # Remove code blocks (```code```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code (`code`)
    text = re.sub(r'`', '', text)
    
    app.logger.info(f"TTS request (after cleaning): {text[:100]}...")

    try:
        # Detect language (if Cyrillic - Russian, otherwise English)
        lang = 'ru' if any('\u0400' <= c <= '\u04FF' for c in text) else 'en'
        
        app.logger.info(f"Using gTTS with language: {lang}")
        
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        app.logger.info(f"gTTS successfully generated audio")
        return send_file(audio_buffer, mimetype="audio/mpeg")
        
    except Exception as e:
        app.logger.exception("Error in gTTS")
        return jsonify({"error": "TTS failed", "details": str(e)}), 500

if __name__ == '__main__':
    # Render всегда задаёт PORT, используем только его
    port_env = os.environ.get('PORT')
    if not port_env:
        raise RuntimeError("Environment variable PORT not set on Render")
    port = int(port_env)
    app.run(host='0.0.0.0', port=port, debug=False)

