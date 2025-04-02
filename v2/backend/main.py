import threading
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from s2s_core import SpeechToSpeechSystem
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import multiprocessing

app = FastAPI()  # ✅ Moved up!

# ✅ Now add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3002", "http://192.168.0.146:3002"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

multiprocessing.freeze_support()
system = SpeechToSpeechSystem()
system.run()

@app.websocket("/ws/state")
async def websocket_state_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while system.running:
            state = 0
            if system.is_speaking:
                state = 2
            elif system.recorder and system.recorder.is_user_speaking:
                state = 1
            await websocket.send_text(str(state))
            await asyncio.sleep(0.25)
    except Exception as e:
        print(f"WebSocket closed: {e}")

@app.post("/update-config")
async def update_config(request: Request):
    try:
        body = await request.json()
        system.set_config(
            system_prompt=body.get("system_prompt"),
            model=body.get("model"),
            voice=body.get("voice"),
            persona=body.get("persona")
        )
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.get("/get-config")
async def get_config():
    return JSONResponse({
        "system_prompt": system.SYSTEM_PROMPT,
        "model": system.llm_model,
        "voice": system.voice_name,
        "persona": system.current_personality
    })


@app.get("/mute-mic")
async def mute_microphone():
    return system.mute_mic()


@app.get("/mute-assistant")
async def mute_assistant():
    return system.mute_ass()


if __name__ == "__main__":
    threading.Thread(target=system.run, daemon=True).start()
    uvicorn.run("speech_web:app", host="0.0.0.0", port=8000, reload=False)