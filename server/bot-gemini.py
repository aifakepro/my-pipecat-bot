# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation."""

import os
import threading                               # ✅ добавлено
from http.server import SimpleHTTPRequestHandler, HTTPServer  # ✅ добавлено

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
from pipecat.runner.types import RunnerArguments
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyParams, DailyTransport

load_dotenv(override=True)

import os
import threading
import socket
from http.server import SimpleHTTPRequestHandler, HTTPServer
import traceback

def keep_alive():
    try:
        port = int(os.environ.get("PORT", 8080))
        addr = ("0.0.0.0", port)
        # проверяем, свободен ли порт
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


# ✅ этот блок добавь ПЕРЕД остальным кодом — он держит Render “живым”
def keep_alive():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    print(f"Keep-alive server running on port {port}")
    server.serve_forever()

threading.Thread(target=keep_alive, daemon=True).start()
# ✅ конец вставки


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


async def run_bot(transport: BaseTransport):
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",
    )

    messages = [
        {
            "role": "user",
            "content": "You are Chatbot, a friendly robot...",
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
    transport = DailyTransport(
        runner_args.room_url,
        runner_args.token,
        "Pipecat Bot",
        params=DailyParams(
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
