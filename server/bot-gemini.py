# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation with LiveKit."""

import os
import threading
import socket
from http.server import SimpleHTTPRequestHandler, HTTPServer
import traceback
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
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams  # Импорт для LiveKit
from pipecat.runner.types import RunnerArguments

load_dotenv(override=True)

# FastAPI для клиентского интерфейса
app = FastAPI()

@app.get("/client")
async def serve_client():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>Pipecat Bot with LiveKit</title><script src="https://unpkg.com/livekit-client@2/dist/livekit-client.umd.js"></script></head>
    <body>
        <video id="localVideo" autoplay muted></video>
        <video id="remoteVideo" autoplay></video>
        <button onclick="joinRoom()">Join Room</button>
        <script>
            const room = new LivekitClient.Room();
            async function joinRoom() {
                await room.connect('YOUR_LIVEKIT_URL', 'YOUR_LIVEKIT_TOKEN', { autoSubscribe: true, publishDefaults: { videoEnabled: true } });
                room.on('trackSubscribed', (track, publication, participant) => {
                    const videoEl = participant.identity === 'bot' ? document.getElementById('remoteVideo') : document.getElementById('localVideo');
                    videoEl.srcObject = new MediaStream([track.mediaStreamTrack]);
                });
            }
        </script>
    </body>
    </html>
    """)

# Запуск Uvicorn
port = int(os.environ.get("PORT", 7860))
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=port), daemon=True).start()

def keep_alive():
    try:
        addr = ("0.0.0.0", port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(addr)
        except Exception as e:
            print("Keep-alive: cannot bind port", addr, " —", e)
            sock.close()
            return
        sock.close()

        server = HTTPServer(addr, SimpleHTTPRequestHandler)
        print(f"Keep-alive server starting on 0.0.0.0:{port}")
        server.serve_forever()
    except Exception:
        print("Keep-alive server failed to start:")
        traceback.print_exc()

threading.Thread(target=keep_alive, daemon=True).start()

sprites = []
script_dir = os.path.dirname(__file__)
for i in range(1, 26):
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

flipped = sprites[::-1]
sprites.extend(flipped)
quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)

class TalkingAnimation(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
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
    await task.queue_frame(quiet_frame)

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.info("Client connected")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    # Генерация токена для LiveKit (нужен для подключения)
    import livekit.api  # Убедись, что livekit-api установлен
    api = livekit.api.AccessToken(
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    ).with_name("Pipecat Bot").with_identity("bot").with_grants(
        livekit.api.VideoGrants(
            room_join=True,
            room=list=True,
            room_admin=True,
            can_publish=True,
            can_subscribe=True,
        )
    ).to_jwt()

    transport = LiveKitTransport(
        url=os.getenv("LIVEKIT_URL", "wss://your-project.livekit.cloud"),  # Твой URL из дашборда
        token=api,  # JWT токен
        room_name="my-room",  # Имя комнаты
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
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
