# Copyright (c) 2024–2025, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Gemini Bot Implementation with LiveKit — Render-ready."""

import os
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from pipecat.runner.types import RunnerArguments
import livekit.api

load_dotenv(override=True)

# Убедимся, что FastAPI использует правильный корень
app = FastAPI()

@app.get("/client")
async def serve_client():
    try:
        client_token = livekit.api.AccessToken(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        ).with_name("Client").with_identity("client").with_grants(
            livekit.api.VideoGrants(
                room_join=True,
                room="my-room",
                can_publish=True,
                can_subscribe=True,
            )
        ).to_jwt()
    except Exception as e:
        logger.error(f"Failed to generate client token: {e}")
        return HTMLResponse(content="<h2>Server misconfigured: check LIVEKIT_API_KEY/SECRET</h2>")

    livekit_url = os.getenv("LIVEKIT_URL", "wss://your-project.livekit.cloud")
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pipecat Bot with LiveKit</title>
        <script src="https://unpkg.com/livekit-client@2/dist/livekit-client.umd.js"></script>
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; }}
            video {{ width: 45%; height: auto; border: 1px solid #ccc; margin: 10px; }}
            button {{ padding: 10px 20px; font-size: 16px; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>Pipecat + LiveKit + Gemini</h1>
        <video id="localVideo" autoplay muted playsinline></video>
        <video id="remoteVideo" autoplay playsinline></video>
        <br>
        <button onclick="joinRoom()">Join Room</button>
        <script>
            const room = new LivekitClient.Room();
            async function joinRoom() {{
                try {{
                    await room.connect('{livekit_url}', '{client_token}', {{
                        autoSubscribe: true,
                        publishDefaults: {{ videoEnabled: true, audioEnabled: true }}
                    }});
                    room.on('trackSubscribed', (track, publication, participant) => {{
                        const videoEl = participant.identity === 'bot' 
                            ? document.getElementById('remoteVideo') 
                            : document.getElementById('localVideo');
                        videoEl.srcObject = new MediaStream([track.mediaStreamTrack]);
                    }});
                    console.log("Connected to room");
                }} catch (e) {{
                    console.error("Join failed:", e);
                    alert("Failed to join room: " + e.message);
                }}
            }}
        </script>
    </body>
    </html>
    """)

# Запуск FastAPI в фоне
port = int(os.environ.get("PORT", 7860))
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"), daemon=True).start()

# Загрузка спрайтов
sprites = []
script_dir = os.path.dirname(__file__)
try:
    for i in range(1, 26):
        full_path = os.path.join(script_dir, "assets", f"robot0{i}.png")
        with Image.open(full_path) as img:
            sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))
    flipped = sprites[::-1]
    sprites.extend(flipped)
    quiet_frame = sprites[0]
    talking_frame = SpriteFrame(images=sprites)
except Exception as e:
    logger.error(f"Failed to load sprites: {e}")
    quiet_frame = None
    talking_frame = None

class TalkingAnimation(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking and talking_frame:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if quiet_frame:
                await self.push_frame(quiet_frame)
            self._is_talking = False
        await self.push_frame(frame, direction)

async def run_bot(transport):
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",
    )

    messages = [
        {
            "role": "user",
            "content": "You are Chatbot, a friendly robot. You can chat with me about anything.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    ta = TalkingAnimation()
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        context_aggregator.user(),
        llm,
        ta,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[RTVIObserver(rtvi)],
    )
    if quiet_frame:
        await task.queue_frame(quiet_frame)

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.info(f"Client connected: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    bot_token = livekit.api.AccessToken(
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    ).with_name("Pipecat Bot").with_identity("bot").with_grants(
        livekit.api.VideoGrants(
            room_join=True,
            room="my-room",
            can_publish=True,
            can_subscribe=True,
        )
    ).to_jwt()

    transport = LiveKitTransport(
        url=os.getenv("LIVEKIT_URL"),
        token=bot_token,
        room_name="my-room",
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=bool(quiet_frame),  # только если есть спрайты
            video_out_width=1024,
            video_out_height=576,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        ),
    )
    await run_bot(transport)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
