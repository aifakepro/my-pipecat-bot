import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.frames.frames import OutputImageRawFrame, SpriteFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.services.small_webrtc import SmallWebRTCTransport, SmallWebRTCTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.runner.types import RunnerArguments
from PIL import Image

app = FastAPI()
app.mount("/prebuilt", SmallWebRTCPrebuiltUI)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/prebuilt/")

# Загрузка спрайтов (опционально)
quiet_frame = None
try:
    img = Image.open("server/assets/robot01.png")
    quiet_frame = OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format)
except:
    pass

async def run_bot(transport):
    llm = GeminiLiveLLMService(api_key=os.getenv("GOOGLE_API_KEY"), voice_id="Puck")
    context = OpenAILLMContext([{"role": "user", "content": "You are a friendly robot."}])
    context_agg = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        context_agg.user(),
        llm,
        transport.output(),
        context_agg.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    if quiet_frame:
        await task.queue_frame(quiet_frame)

    runner = PipelineRunner()
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    transport = SmallWebRTCTransport(
        params=SmallWebRTCTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=bool(quiet_frame),
            vad_analyzer=SileroVADAnalyzer(),
        )
    )
    await run_bot(transport)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
