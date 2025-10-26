from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import requests
import os
import re
from io import BytesIO
import google.generativeai as genai
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# Настройка API ключей из окружения
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
    """Транскрибує аудіо за допомогою Deepgram API"""
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
    """Отримує відповідь від Gemini AI"""
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        app.logger.warning("Нет текста в запросе /chat")
        return jsonify({"error": "No text provided"}), 400

    user_text = data.get('text', '')
    language = data.get('language', 'ru')
    app.logger.info(f"Запрос к Gemini: {user_text}, язык: {language}")

    if not GEMINI_API_KEY:
        app.logger.error("Gemini API key не настроен")
        return jsonify({"error": "Gemini API key not configured"}), 500

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Системний промпт залежно від мови
        prompts = {
            'uk': """Ти голосовий асистент. Дотримуйся цих правил:
- Відповідай УКРАЇНСЬКОЮ мовою
- Відповідай коротко і по суті (1-3 речення для простих питань)
- Уникай надмірного форматування markdown
- Говори природно, як жива людина
- Не використовуй списки без потреби
- Будь ввічливим та дружнім""",
            'ru': """Ты голосовой ассистент. Следуй этим правилам:
- Отвечай на РУССКОМ языке
- Отвечай кратко и по сути (1-3 предложения для простых вопросов)
- Избегай излишнего форматирования markdown
- Говори естественно, как живой человек
- Не используй списки без необходимости
- Будь вежливым и дружелюбным""",
            'en': """You are a voice assistant. Follow these rules:
- Respond in ENGLISH
- Answer briefly and to the point (1-3 sentences for simple questions)
- Avoid excessive markdown formatting
- Speak naturally, like a real person
- Don't use lists unless necessary
- Be polite and friendly"""
        }
        
        system_instruction = prompts.get(language, prompts['ru'])
        
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=system_instruction
        )

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
    """Перетворює текст на аудіо за допомогою gTTS"""
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        app.logger.error("Нет текста в запросе /speak")
        return jsonify({"error": "No text provided"}), 400

    text = data.get('text', '').strip()
    language = data.get('language', 'ru')
    
    if not text:
        app.logger.error("Текст пустой после strip()")
        return jsonify({"error": "Empty text provided"}), 400
    
    # Очищаємо текст від markdown форматування та зайвих символів для озвучки
    text = re.sub(r'\*', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'_', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`', '', text)
    text = text.replace('"', '').replace('"', '').replace('„', '')
    text = text.replace("'", '').replace("'", '').replace('«', '').replace('»', '')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(http[s]?://[^\)]+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    app.logger.info(f"Запрос на озвучку (после очистки): {text[:100]}..., язык: {language}")

    try:
        if language not in ['uk', 'ru', 'en']:
            language = 'ru' if any('\u0400' <= c <= '\u04FF' for c in text) else 'en'
        
        app.logger.info(f"Используем gTTS с языком: {language}")
        
        tts = gTTS(text=text, lang=language, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        app.logger.info(f"gTTS успешно сгенерировал аудио")
        return send_file(audio_buffer, mimetype="audio/mpeg")
        
    except Exception as e:
        app.logger.exception("Ошибка в gTTS")
        return jsonify({"error": "TTS failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
