from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from s2s_core import SpeechToSpeechSystem

app = FastAPI()
speech_system = SpeechToSpeechSystem()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/realtime-sts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # State: user is speaking
            await websocket.send_text(json.dumps({"type": "state", "value": 1}))

            while speech_system.is_speaking or speech_system.is_initializing_tts:
                await asyncio.sleep(0.05)

            # State: assistant is responding
            await websocket.send_text(json.dumps({"type": "state", "value": 2}))

            loop = asyncio.get_event_loop()
            audio_chunks = await loop.run_in_executor(None, lambda: list(speech_system.process_transcription_api(data)))

            for chunk in audio_chunks:
                await websocket.send_bytes(chunk)

            await websocket.send_text(json.dumps({"type": "state", "value": 0}))
            await websocket.send_text("__END__")

    except Exception as e:
        print(f"🔥 WebSocket error: {e}")
        await websocket.send_text(json.dumps({"type": "state", "value": 0}))

@app.post("/configure")
async def configure_backend(config: dict):
    speech_system.set_config(
        system_prompt=config.get("system_prompt"),
        model=config.get("model"),
        persona=config.get("persona"),
        voice=config.get("voice")
    )
    return {"status": "ok"}