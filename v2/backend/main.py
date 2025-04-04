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
    allow_origins=["http://localhost:3002", "http://192.168.0.146:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

multiprocessing.freeze_support()
system = SpeechToSpeechSystem()

@app.websocket("/ws/state")
async def websocket_state_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while system.running:
            state = "none"
            if system.is_speaking:
                state = "assistant"
            elif system.is_speaking == False:
                state = "user"
            await websocket.send_text(state)
            await asyncio.sleep(0.25)
    except Exception as e:
        print(f"WebSocket closed: {e}")

@app.post("/update-config")
async def update_config(request: Request):
    try:
        body = await request.json()
        print("🔍 Received config:", body)
        system.restart(
            personality=body.get("persona"),
            syst=body.get("system_prompt") or None,
            voice=body.get("voice"),
            llm=body.get("model")
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
    result = system.mute_mic()
    return JSONResponse(result)


@app.get("/mute-assistant")
async def mute_assistant():
    result = system.mute_ass()
    return JSONResponse(result)


if __name__ == "__main__":
    try:
        def background_worker():
            print("🟢 system.run() started in background thread")
            system.run()
        threading.Thread(target=background_worker, daemon=True).start()
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("🛑 KeyboardInterrupt received! Stopping system...")
        system.stop()