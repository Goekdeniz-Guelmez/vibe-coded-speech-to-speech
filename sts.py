import threading
import time
import queue
import pyaudio
from openai import OpenAI
from RealtimeSTT import AudioToTextRecorder
import datetime
import numpy as np
import concurrent.futures
import difflib

now = datetime.datetime.now()

SYSTEM = f"""You are a helpful AI assistant, and designed for a real-time speech-to-speech system. Your responses will be directly converted to spoken audio. Therefore, it is critical that your output is clean and easily understandable when spoken aloud.

Adhere to the following guidelines:
*   **No Emojis or Symbols:** Do not include any emojis, symbols, or special characters in your responses.
*   **Plain Text Only:**  Your responses should be in plain text. Avoid Markdown formatting, LaTeX, or any other markup languages.
*   **Concise and Clear Language:** Use clear, concise language that is easy to understand when spoken. Avoid complex sentence structures or jargon.
*   **Direct and Conversational Tone:** Maintain a direct and conversational tone, as if you were speaking directly to a person.
*   **Focus on Information Delivery:** Prioritize delivering information clearly and efficiently.

Current time: {now.strftime("%H:%M")}
Current date: {now.strftime("%Y-%m-%d")}
Current day: {now.strftime("%A")}"""

class SpeechToSpeechSystem:
    def __init__(self):
        # Initialize audio parameters
        self.TTS_RATE = 24000  # For TTS output
        
        # Initialize Ollama client
        self.ollama_client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',
        )
        
        # Initialize TTS client
        self.tts_client = OpenAI(
            base_url="http://localhost:8880/v1", 
            api_key="not-needed"
        )
        
        # Initialize PyAudio for TTS output
        self.pyaudio = pyaudio.PyAudio()
        self.tts_stream = None
        
        # Initialize RealtimeSTT recorder with modified parameters
        self.recorder = AudioToTextRecorder(post_speech_silence_duration=0.6)
        
        # System prompt for Ollama
        self.SYSTEM_PROMPT = SYSTEM
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Flag to control system
        self.running = True
        
        # Flag to indicate if the system is currently speaking
        self.is_speaking = False
        
        # Cooldown period after system speaks (in seconds) - increased from 0.3
        self.cooldown_period = 1.5
        self.last_spoke_time = 0
        
        # Sentence boundary detection for chunking responses
        self.sentence_end_markers = ['.', '!', '?']
        self.pause_markers = [',', ';', ':']
        
        # Audio queue for smooth playback
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        
        # Thread pool for parallel TTS processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Overlap parameters
        self.overlap_duration = 0.02  # 30ms overlap between chunks
        self.overlap_samples = int(self.TTS_RATE * self.overlap_duration)
        
        # Store the last chunk's ending for overlapping
        self.last_chunk_ending = None
        
        # Store recent responses to detect echo/feedback
        self.recent_responses = []
        self.max_recent_responses = 5
        
        # Pause listening during TTS initialization
        self.is_initializing_tts = False
        
    def is_similar_to_recent_response(self, transcription):
        """Check if the transcription is similar to any recent system response"""
        if not transcription:
            return False
            
        # Clean the transcription for comparison
        clean_transcription = transcription.lower().strip()
        
        for response in self.recent_responses:
            clean_response = response.lower().strip()
            
            # Check for exact match
            if clean_transcription == clean_response:
                return True
                
            # Check for high similarity using difflib
            similarity = difflib.SequenceMatcher(None, clean_transcription, clean_response).ratio()
            if similarity > 0.7:  # 70% similarity threshold
                return True
                
            # Check if transcription is contained within response
            if len(clean_transcription) > 5 and clean_transcription in clean_response:
                return True
                
        return False
        
    def process_transcription(self, transcription):
        """Process transcribed speech and generate response"""
        # Skip processing if the system is speaking, in cooldown period, or initializing TTS
        current_time = time.time()
        if (self.is_speaking or 
            (current_time - self.last_spoke_time < self.cooldown_period) or
            self.is_initializing_tts):
            return
            
        # Avoid processing empty transcriptions
        if not transcription or not transcription.strip():
            return
            
        # Check if this is the system hearing itself
        if self.is_similar_to_recent_response(transcription):
            print(f"Ignored self-echo: {transcription}")
            return
            
        print(f"Transcribed: {transcription}")
        
        try:
            # Set speaking flag to prevent processing our own speech
            self.is_speaking = True
            self.is_initializing_tts = True
            
            # Reset the last chunk ending
            self.last_chunk_ending = None
            
            # Generate response using Ollama
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": transcription}
            ]
            
            print("Generating response...")
            
            # Clear any existing audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
            
            # Initialize the audio stream immediately
            if self.tts_stream is None:
                self.tts_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.TTS_RATE,
                    output=True,
                    frames_per_buffer=512  # Smaller buffer for lower latency
                )
            
            # Start the playback thread if not already running
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.start_audio_playback_thread()
            
            # TTS initialization complete
            self.is_initializing_tts = False
            
            stream = self.ollama_client.chat.completions.create(
                model='gemma3-tools',
                messages=messages,
                stream=True,
                temperature=0.7
            )
            
            full_response = ""
            current_chunk = ""
            future_to_chunk = {}
            chunk_count = 0
            
            # Process each chunk as it arrives
            for chunk in stream:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        current_chunk += content
                        
                        # Process chunks at natural break points or when they get long enough
                        if self.should_process_chunk(current_chunk):
                            # Submit TTS task to thread pool
                            future = self.executor.submit(
                                self.generate_speech, 
                                current_chunk.strip(), 
                                chunk_count == 0
                            )
                            future_to_chunk[future] = current_chunk.strip()
                            current_chunk = ""
                            chunk_count += 1
                            
                            # Check if any futures are done to queue them for playback
                            self.check_and_queue_completed_futures(future_to_chunk)
            
            # Process any remaining text
            if current_chunk:
                future = self.executor.submit(
                    self.generate_speech, 
                    current_chunk.strip(), 
                    chunk_count == 0
                )
                future_to_chunk[future] = current_chunk.strip()
            
            # Wait for all pending TTS tasks to complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    audio_bytes, is_last = future.result()
                    if audio_bytes:
                        self.audio_queue.put(("AUDIO", (audio_bytes, is_last)))
                except Exception as e:
                    print(f"Error processing TTS: {e}")
            
            # Signal end of response
            self.audio_queue.put(("END_OF_RESPONSE", None))
            
            print(f"Full response: {full_response}")
            
            # Store this response for echo detection
            with self.lock:
                self.recent_responses.append(full_response)
                # Keep only the most recent responses
                if len(self.recent_responses) > self.max_recent_responses:
                    self.recent_responses.pop(0)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Signal end of response on error
            self.audio_queue.put(("END_OF_RESPONSE", None))
        finally:
            # Reset speaking flag will be set to False after all audio has been played
            self.last_spoke_time = time.time()
            print("Listening again...")
    
    def check_and_queue_completed_futures(self, future_to_chunk):
        """Check for completed futures and queue them for playback"""
        done_futures = []
        for future in list(future_to_chunk.keys()):
            if future.done():
                try:
                    audio_bytes, is_last = future.result()
                    if audio_bytes:
                        self.audio_queue.put(("AUDIO", (audio_bytes, is_last)))
                    done_futures.append(future)
                except Exception as e:
                    print(f"Error processing TTS: {e}")
                    done_futures.append(future)
        
        # Remove processed futures
        for future in done_futures:
            future_to_chunk.pop(future, None)

    def should_process_chunk(self, text):
        """Determine if a chunk should be processed based on content and length"""
        # Process if we have a complete sentence
        if any(text.endswith(marker) for marker in self.sentence_end_markers):
            return True
            
        # Process if we have a natural pause and enough text
        if len(text) > 40 and any(text.endswith(marker) for marker in self.pause_markers):
            return True
            
        # Process if chunk is getting too long
        if len(text) > 100:
            # Find the last space to break at a word boundary
            last_space = text.rfind(' ')
            if last_space > 20:  # Only break at word if we have enough content
                return True
                
        # Otherwise, keep accumulating
        return False

    def generate_speech(self, text_chunk, is_first=False):
        """Generate speech for text and return audio bytes"""
        if not text_chunk.strip():
            return None, False
            
        try:
            print(f"Generating TTS for: {text_chunk}")
            
            # Generate speech for the text chunk
            with self.tts_client.audio.speech.with_streaming_response.create(
                model="kokoro",
                voice="af_bella",
                response_format="pcm",
                input=text_chunk
            ) as response:
                # Collect all audio data for this chunk
                audio_data = b''
                for chunk in response.iter_bytes(chunk_size=4096):
                    audio_data += chunk
                
                # Convert to numpy array for processing
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Trim silence from the beginning of non-first chunks for smoother transitions
                if not is_first:
                    # Find the first non-silent sample (threshold of 0.005)
                    threshold = 0.005
                    start_idx = 0
                    for i in range(min(500, len(audio_np))):  # Only check first 500 samples
                        if abs(audio_np[i]) > threshold:
                            start_idx = max(0, i - 3)  # Keep a tiny bit of lead-in
                            break
                    audio_np = audio_np[start_idx:]
                
                # Check if this chunk ends with a sentence marker
                is_sentence_end = any(text_chunk.endswith(marker) for marker in self.sentence_end_markers)
                
                # Use different overlap settings based on chunk type
                overlap_samples = self.overlap_samples
                if not is_sentence_end:
                    # For mid-sentence chunks, use almost zero pause (just enough for smooth transition)
                    overlap_samples = int(self.TTS_RATE * 0.01)  # 10ms overlap for mid-sentence chunks
                
                # Store this chunk's ending for future overlapping
                chunk_ending = None
                if len(audio_np) > overlap_samples:
                    chunk_ending = audio_np[-overlap_samples:].copy()
                
                # Apply overlapping with previous chunk if available
                with self.lock:
                    if not is_first and self.last_chunk_ending is not None:
                        # Create crossfade between chunks
                        fade_in = np.linspace(0, 1, len(self.last_chunk_ending))
                        fade_out = np.linspace(1, 0, len(self.last_chunk_ending))
                        
                        # Apply crossfade to the beginning of current chunk
                        overlap_region = min(len(self.last_chunk_ending), len(audio_np))
                        for i in range(overlap_region):
                            if i < len(audio_np):
                                audio_np[i] = (audio_np[i] * fade_in[i]) + (self.last_chunk_ending[i] * fade_out[i])
                    
                    # Update last chunk ending for next time
                    self.last_chunk_ending = chunk_ending
                
                # Convert back to int16 bytes for playback
                audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
                
                # Return the audio bytes and a flag indicating if this is the last chunk
                return audio_bytes, is_sentence_end
            
        except Exception as e:
            print(f"Error in TTS generation: {e}")
            return None, False
    
    def start_audio_playback_thread(self):
        """Start a thread for continuous audio playback"""
        self.playback_thread = threading.Thread(target=self.audio_playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def audio_playback_worker(self):
        """Worker thread for continuous audio playback"""
        try:
            # Process audio chunks from the queue in order
            while self.running:
                try:
                    # Get the next item from the queue with a timeout
                    item_type, data = self.audio_queue.get(timeout=0.05)
                    
                    if item_type == "AUDIO" and data and self.tts_stream:
                        audio_bytes, is_first = data
                        
                        # Add a very small pause between chunks for clarity
                        if not is_first:
                            time.sleep(0.01)  # 10ms pause between chunks
                            
                        # Play the audio chunk
                        self.tts_stream.write(audio_bytes)
                        
                    elif item_type == "END_OF_RESPONSE":
                        # End of response reached
                        # Wait a short time to ensure audio is fully played
                        time.sleep(0.2)
                        # Set speaking flag to false
                        self.is_speaking = False
                        # Update the last spoke time to ensure cooldown
                        self.last_spoke_time = time.time()
                        break
                        
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # Queue is empty but we're still processing the response
                    continue
                    
        except Exception as e:
            print(f"Error in audio playback worker: {e}")
        finally:
            self.is_speaking = False
            self.last_spoke_time = time.time()
    
    def run(self):
        """Start the speech-to-speech system"""
        try:
            print("Speech-to-Speech system is running. Press Ctrl+C to stop.")
            print("Wait until RealtimeSTT says 'speak now'")
            
            # Main loop
            while self.running:
                try:
                    # Only listen when not speaking or in cooldown
                    if not self.is_speaking and (time.time() - self.last_spoke_time > self.cooldown_period) and not self.is_initializing_tts:
                        # This will call process_transcription when speech is detected and transcribed
                        self.recorder.text(self.process_transcription)
                    else:
                        # Small sleep to prevent CPU hogging during speaking/cooldown
                        time.sleep(0.05)
                except Exception as e:
                    print(f"Error in speech recording: {e}")
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            print("Stopping...")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        # Clean up PyAudio
        if self.tts_stream:
            self.tts_stream.stop_stream()
            self.tts_stream.close()
            self.tts_stream = None
        self.pyaudio.terminate()
        self.executor.shutdown()
        print("System stopped.")

# Run the system
if __name__ == "__main__":
    system = SpeechToSpeechSystem()
    system.run()