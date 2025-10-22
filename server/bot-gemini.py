import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import httpx
import google.generativeai as genai
from openai import OpenAI

load_dotenv()

app = FastAPI()

# Настройка
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # для Whisper

# STT: Whisper
def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"  # или .wav
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        return transcript.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

# LLM: Gemini
def get_gemini_response(text: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"You are a friendly robot. Answer briefly: {text}"
        )
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")

# TTS: ElevenLabs
async def text_to_speech(text: str) -> bytes:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
                headers={"xi-api-key": elevenlabs_key},
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
                }
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="TTS failed")
            return resp.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

# Эндпоинт: аудио → текст → LLM → аудио
@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio")

    # 1. Получить байты
    audio_bytes = await audio.read()

    # 2. STT
    user_text = transcribe_audio(audio_bytes)
    print("User:", user_text)

    # 3. LLM
    bot_text = get_gemini_response(user_text)
    print("Bot:", bot_text)

    # 4. TTS
    audio_response = await text_to_speech(bot_text)

    # 5. Отдать аудио
    return StreamingResponse(
        io.BytesIO(audio_response),
        media_type="audio/mpeg"
    )
