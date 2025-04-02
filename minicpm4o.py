from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import soundfile as sf
import base64
import io
import librosa
import signal
import sys
import gc
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=10, ping_interval=5)

# Global model and session storage with lock
model = None
tokenizer = None
sessions = {}
model_lock = threading.Lock()
session_lock = threading.Lock()
cleanup_event = threading.Event()
device = torch.device("mps" if torch.has_mps else "cpu")

# HTML page with embedded Socket.IO client
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>MiniCPM-o Voice Chat</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            text-align: center;
        }
        button {
            padding: 10px 20px;
            margin: 10px;
            font-size: 16px;
            cursor: pointer;
        }
        #status, #debug {
            margin: 20px;
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f0f0;
        }
        #debug {
            text-align: left;
            font-family: monospace;
            font-size: 12px;
            height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MiniCPM-o Voice Chat</h1>
        <div>
            <button id="startButton">Start Conversation</button>
            <button id="stopButton" disabled>Stop Conversation</button>
        </div>
        <div id="status">Not connected</div>
        <div id="debug">Debug logs will appear here...</div>
        <audio id="outputAudio"></audio>
    </div>

    <script>
        let socket;
        let mediaRecorder;
        let isRecording = false;
        const debug = document.getElementById('debug');
        
        function log(msg) {
            console.log(msg);
            debug.innerHTML += msg + '<br>';
            debug.scrollTop = debug.scrollHeight;
        }
        
        function updateStatus(message) {
            document.getElementById('status').textContent = message;
            log(message);
        }

        async function startRecording() {
            try {
                updateStatus('Requesting microphone access...');
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000,
                        channelCount: 1
                    }
                });
                updateStatus('Microphone access granted');
                
                const audioContext = new AudioContext({
                    sampleRate: 16000
                });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (isRecording) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const wavBuffer = audioBufferToWav(inputData, 16000);
                        const base64Audio = btoa(
                            String.fromCharCode.apply(null, new Uint8Array(wavBuffer))
                        );
                        socket.emit('stream_chunk', { audio: base64Audio });
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                document.getElementById('startButton').disabled = true;
                document.getElementById('stopButton').disabled = false;
                
            } catch (err) {
                console.error('Error:', err);
                updateStatus('Error: ' + err.message);
            }
        }

        function audioBufferToWav(audioData, sampleRate) {
            const buffer = new ArrayBuffer(44 + audioData.length * 4);
            const view = new DataView(buffer);
            
            // Write WAV header
            const writeString = (view, offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };
            
            const numChannels = 1;
            const bitDepth = 32;
            const dataLength = audioData.length * 4;
            
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + dataLength, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM format
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
            view.setUint16(32, numChannels * (bitDepth / 8), true);
            view.setUint16(34, bitDepth, true);
            writeString(view, 36, 'data');
            view.setUint32(40, dataLength, true);
            
            // Write audio data
            const offset = 44;
            for (let i = 0; i < audioData.length; i++) {
                view.setFloat32(offset + (i * 4), audioData[i], true);
            }
            
            return buffer;
        }

        function stopRecording() {
            if (isRecording) {
                isRecording = false;
            }
            document.getElementById('startButton').disabled = false;
            document.getElementById('stopButton').disabled = true;
            updateStatus('Stopped recording');
        }

        function connectSocket() {
            socket = io(window.location.origin);

            socket.on('connect', () => {
                log('Socket connected');
                updateStatus('Connected to server');
            });

            socket.on('disconnect', () => {
                log('Socket disconnected');
                updateStatus('Disconnected from server');
                stopRecording();
            });

            socket.on('error', (data) => {
                console.error('Server error:', data.message);
                updateStatus('Error: ' + data.message);
            });

            socket.on('response', (data) => {
                if (data.text) {
                    log('Received text: ' + data.text);
                }
                
                if (data.audio) {
                    log('Received audio response');
                    const audio = new Audio('data:audio/wav;base64,' + data.audio);
                    audio.play().catch(e => log('Error playing audio: ' + e));
                }
            });
        }

        document.getElementById('startButton').onclick = () => {
            if (!socket || !socket.connected) {
                connectSocket();
            }
            startRecording();
        };

        document.getElementById('stopButton').onclick = stopRecording;

        window.onload = () => {
            if (!window.isSecureContext) {
                updateStatus('Warning: Page not in secure context. Microphone access may be restricted.');
            }
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                updateStatus('Error: Your browser does not support audio input');
            }
        };
    </script>
</body>
</html>
"""

def validate_audio_data(audio_data):
    """Validate audio data before processing"""
    if not audio_data:
        raise ValueError("Empty audio data received")
    if len(audio_data) < 44:  # Minimum WAV header size
        raise ValueError("Audio data too short to be valid")
    return True

def process_audio_chunk(audio_data, expected_sr=16000):
    """Process audio chunk with enhanced error handling"""
    try:
        # Read as WAV
        audio_buffer = io.BytesIO(audio_data)
        try:
            audio_np, sr = sf.read(audio_buffer)
        except Exception as wav_error:
            print(f"WAV reading failed: {wav_error}, attempting PCM decode")
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            sr = expected_sr

        # Ensure mono
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)

        # Resample if necessary
        if sr != expected_sr:
            audio_np = librosa.resample(y=audio_np, orig_sr=sr, target_sr=expected_sr)

        # Normalize
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np))

        # Validate
        if np.isnan(audio_np).any() or np.isinf(audio_np).any():
            raise ValueError("Invalid values in processed audio")

        return audio_np.astype(np.float32), expected_sr

    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")

def initialize_model():
    """Initialize model with error handling"""
    global model, tokenizer, system_prompt
    try:
        print("Initializing model...")
        model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.float32,
            init_vision=False,
            init_audio=True,
            init_tts=True
        )
        model = model.eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)
        model.init_tts()

        # Add your system prompt here
        system_prompt = model.get_sys_prompt(mode='audio', language='en')
        print("Model initialized successfully on device:", device)
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        sys.exit(1)

def cleanup_sessions():
    """Clean up inactive sessions"""
    while not cleanup_event.is_set():
        try:
            with session_lock:
                current_time = time.time()
                to_remove = []
                for sid, session in sessions.items():
                    if current_time - session.get('last_active', current_time) > 300:
                        to_remove.append(sid)
                for sid in to_remove:
                    cleanup_session(sid)
                    print(f"Cleaned up inactive session: {sid}")
        except Exception as e:
            print(f"Error in session cleanup: {str(e)}")
        time.sleep(60)

def cleanup_session(sid):
    """Clean up a specific session"""
    try:
        if sid in sessions:
            with session_lock:
                if model:
                    model.reset_session()
                del sessions[sid]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error cleaning up session {sid}: {str(e)}")

@app.route('/')
def index():
    """Serve the HTML page"""
    return HTML_PAGE

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    try:
        sid = request.sid
        with session_lock:
            if model is None:
                emit('error', {'message': 'Server initializing, please try again'})
                disconnect()
                return
            
            sessions[sid] = {
                'session_id': f'session_{sid}',
                'chunks': [],
                'processing': False,
                'initialized': False,
                'last_active': time.time()
            }
            model.reset_session()
        print(f'Client connected: {sid}')
    except Exception as e:
        print(f"Error in handle_connect: {str(e)}")
        disconnect()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    try:
        sid = request.sid
        cleanup_session(sid)
        print(f'Client disconnected: {sid}')
    except Exception as e:
        print(f"Error in handle_disconnect: {str(e)}")

@socketio.on('stream_chunk')
def handle_chunk(data):
    """Handle incoming audio chunk"""
    try:
        sid = request.sid
        if sid not in sessions:
            emit('error', {'message': 'Invalid session'})
            disconnect()
            return
        
        with session_lock:
            session = sessions[sid]
            session['last_active'] = time.time()
            
            try:
                audio_data = base64.b64decode(data['audio'])
                validate_audio_data(audio_data)
            except Exception as e:
                emit('error', {'message': f'Invalid audio data: {str(e)}'})
                return
            
            try:
                audio_np, sr = process_audio_chunk(audio_data)
                if audio_np is None:
                    emit('error', {'message': 'Audio processing failed'})
                    return
            except Exception as e:
                emit('error', {'message': f'Audio processing error: {str(e)}'})
                return
            
            session['chunks'].append(audio_np)
            
            if not session['processing']:
                session['processing'] = True
                try:
                    with model_lock:
                        model.reset_session()
                        
                        sys_msg = model.get_sys_prompt(mode='audio', language='en')
                        model.streaming_prefill(
                            session_id=session['session_id'],
                            msgs=[sys_msg], 
                            tokenizer=tokenizer
                        )
                        
                        for chunk in session['chunks']:
                            model.streaming_prefill(
                                session_id=session['session_id'],
                                msgs=[{"role": "user", "content": [chunk]}],
                                tokenizer=tokenizer
                            )
                        
                        audios = []
                        text = ""
                        
                        res = model.streaming_generate(
                            session_id=session['session_id'],
                            tokenizer=tokenizer,
                            temperature=0.5,
                            generate_audio=True
                        )
                        
                        for r in res:
                            if hasattr(r, 'audio_wav'):
                                audios.append(r.audio_wav)
                                text += r.text if hasattr(r, 'text') else ''
                        
                        if audios:
                            final_audio = np.concatenate(audios)
                            output_buffer = io.BytesIO()


                            sf.write(output_buffer, final_audio, r.sampling_rate, format='WAV')
                            audio_base64 = base64.b64encode(output_buffer.getvalue()).decode()
                            
                            emit('response', {
                                'text': text,
                                'audio': audio_base64
                            })
                        
                        session['chunks'] = []
                        
                except Exception as e:
                    print(f"Error in processing: {str(e)}")
                    model.reset_session()
                    session['chunks'] = []
                finally:
                    session['processing'] = False

    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        emit('error', {'message': str(e)})
        cleanup_session(sid)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Received signal {signum}")
    cleanup_event.set()
    sys.exit(0)

if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start session cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_sessions)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Run server
    print("Server running on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
                

