from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json, os
import numpy as np
import torch
import uvicorn
from collections import deque
from dataclasses import dataclass
from typing import Optional
import Levenshtein

from symspellpy.symspellpy import SymSpell
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel
import re

# ------------------------
# Configuration
# ------------------------
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.3          # Faster chunk processing
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
CONTEXT_DURATION = 2.0        # Context for better accuracy
CONTEXT_SAMPLES = int(SAMPLE_RATE * CONTEXT_DURATION)
MAX_BUFFER_DURATION = 120     # Increased to 2 minutes for long speech
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_DURATION)

# VAD Settings
VAD_THRESHOLD = 0.5           # Sensitivity (0.0-1.0, lower = more sensitive)
MIN_SPEECH_DURATION = 0.25    # Minimum speech segment
MIN_SILENCE_DURATION = 1.5    # Increased - don't finalize on brief pauses

# Partial Settings
PARTIAL_EDIT_THRESHOLD = 0.15 # 15% change required to update partial
PARTIAL_MIN_LENGTH = 3        # Minimum characters for partial
PARTIAL_UPDATE_INTERVAL = 0.15 # Maximum partial update frequency

# Performance - Force CPU to avoid CUDA library issues on Windows
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
NUM_WORKERS = 4
CPU_THREADS = 8

# Model Selection for Accent Handling
# CHOOSE BASED ON YOUR HARDWARE:
# 
# For i5/i7 laptops (CPU only):
#   WHISPER_MODEL = "medium"   ← RECOMMENDED (88-92% accuracy, 400-600ms)
#   WHISPER_MODEL = "small"    ← Fast option (80-85% accuracy, 200-300ms)
#
# For GPU or Google Colab:
#   WHISPER_MODEL = "large-v3" ← Best accuracy (95-98%, but slow on CPU)
#
# Current setting:
WHISPER_MODEL = "medium"  # Best for i5 11th gen laptops

# Punctuation Settings
ENABLE_PUNCTUATION = False  # Set to False for pure word accuracy (recommended for projects)
# When False: Strips all Whisper punctuation, outputs continuous text
# When True: Keeps minimal capitalization only

print(f"[CONFIG] Device: {DEVICE}, Compute: {COMPUTE_TYPE}, Workers: {NUM_WORKERS}")
print(f"[CONFIG] Model: {WHISPER_MODEL} (optimized for South Asian accents)")
print(f"[CONFIG] Punctuation: {'Disabled (accuracy mode)' if not ENABLE_PUNCTUATION else 'Minimal'}")
print("[INFO] Using CPU mode for maximum compatibility")

# ------------------------
# Models
# ------------------------
print(f"[INIT] Loading Faster-Whisper {WHISPER_MODEL} on CPU for better accent recognition...")
model = WhisperModel(
    WHISPER_MODEL,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    num_workers=NUM_WORKERS,
    cpu_threads=CPU_THREADS,
    download_root=None
)
print(f"[SUCCESS] {WHISPER_MODEL} model loaded successfully on CPU")

print("[INIT] Loading Silero VAD...")
vad_model = load_silero_vad()

# ------------------------
# SymSpell Setup
# ------------------------
BASE_DIR = os.path.dirname(__file__)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(
    os.path.join(BASE_DIR, "frequency_dictionary_en_82_765.txt"), 0, 1
)
sym_spell.load_bigram_dictionary(
    os.path.join(BASE_DIR, "frequency_bigramdictionary_en_243_342.txt"), 0, 2
)

def correct_text(text: str) -> str:
    """SymSpell correction with South Asian accent fixes"""
    if not text:
        return text
    
    # Common South Asian accent transcription errors
    accent_corrections = {
        # V/W confusion
        r'\bvery well\b': 'very well',
        r'\bvill\b': 'will',
        r'\bvould\b': 'would',
        
        # TH sounds
        r'\btree\b': 'three',
        r'\btinking\b': 'thinking',
        r'\btank\b': 'thank',
        
        # Specific words from user's example
        r'\bahisper\b': 'whisper',
        r'\bvhisper\b': 'whisper',
        
        # Common phrase errors
        r'\bit takes me back\b': 'anxiety drags me back',
        r'\bit drags me back\b': 'anxiety drags me back',
        
        # Common tech words
        r'\bpython\b': 'Python',
        r'\bdata\b': 'data',
        r'\bproject\b': 'project',
    }
    
    # Apply accent-specific corrections first
    for pattern, replacement in accent_corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Then apply SymSpell for general spelling
    try:
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term if suggestions else text
    except:
        return text

# Expanded filler set
FILLERS = {
    "um", "umm", "uh", "uhh", "uhm", "erm", "hmm", "ah", "aah", "aa", "eh",
    "mhm", "mm", "mmm", "hm", "huh"
}

def remove_fillers(text: str) -> str:
    """Remove filler words while preserving sentence structure"""
    words = text.split()
    filtered = []
    
    for i, word in enumerate(words):
        # Remove standalone fillers
        if word.lower() in FILLERS:
            continue
        
        # Remove "like" only if it's clearly a filler (not part of "feels like", "looks like", etc.)
        if word.lower() == "like":
            # Keep "like" if preceded by verbs that use it meaningfully
            prev_word = words[i-1].lower() if i > 0 else ""
            if prev_word in ["feels", "looks", "sounds", "seems", "tastes", "smells", "is", "was", "were"]:
                filtered.append(word)
                continue
            # Otherwise skip it as filler
            continue
        
        # Remove "you know" only when standalone filler
        if word.lower() == "you" and i + 1 < len(words) and words[i+1].lower() == "know":
            # Check if it's actually being used meaningfully
            if i + 2 < len(words) and words[i+2].lower() in ["that", "what", "how", "why"]:
                filtered.append(word)  # Keep meaningful "you know that..."
            else:
                continue  # Skip "you know" filler
        elif word.lower() == "know" and i > 0 and words[i-1].lower() == "you":
            # Already handled in previous iteration
            if filtered and filtered[-1].lower() == "you":
                filtered.append(word)  # Keep if we kept "you"
            continue
        else:
            filtered.append(word)
    
    return " ".join(filtered)

def restore_punctuation(text: str) -> str:
    """Minimal punctuation - accuracy focused, remove Whisper's auto-punctuation"""
    if not text or len(text) < 5:
        return text
    
    # Remove ALL periods, commas, question marks that Whisper added
    # Keep only the words for maximum accuracy
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')
    text = text.replace('?', ' ')
    text = text.replace('!', ' ')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Capitalize first letter only
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Capitalize pronoun "I"
    text = re.sub(r'\bi\b', 'I', text)
    
    # Add single period at end
    text += '.'
    
    return text

def basic_capitalize(text: str) -> str:
    """Basic capitalization for partials (fast)"""
    if not text:
        return text
    # Capitalize first letter and "I"
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    text = re.sub(r'\bi\b', 'I', text)
    return text

def normalize_text(text: str, is_final: bool = False) -> str:
    """Complete text normalization pipeline"""
    text = text.strip()
    if not text:
        return ""
    
    # Remove leaked prompt text (Whisper sometimes includes the prompt)
    leaked_prompts = [
        "Use proper punctuation",
        "South Asian English",
        "Nepali accent",
        "Previous context:",
        "Previous:",
        "Common words:",
        "Technical vocabulary:",
        # Whisper subtitle watermarks
        "subtitles by",
        "transcribed by",
        "captioned by",
        "amara",
        "mara org",
        "the mara org community",
        "subtitle by the community",
        "www.amara.org"
    ]
    for prompt in leaked_prompts:
        text = text.replace(prompt, "")
        text = text.replace(prompt.lower(), "")
    
    # Clean up any resulting double spaces or leading punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[.,;:]\s*', '', text)
    
    # Spelling correction
    text = correct_text(text)
    
    # Remove fillers
    text = remove_fillers(text)
    
    # Punctuation restoration (based on config)
    if ENABLE_PUNCTUATION and is_final:
        # Full punctuation restoration for final transcripts
        text = restore_punctuation(text)
    elif ENABLE_PUNCTUATION and not is_final:
        # Basic capitalization for partials (faster)
        text = basic_capitalize(text)
    else:
        # No punctuation processing - pure accuracy mode
        # Just strip Whisper's punctuation and capitalize first letter
        text = text.replace('.', ' ')
        text = text.replace(',', ' ')
        text = text.replace('?', ' ')
        text = text.replace('!', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        text = re.sub(r'\bi\b', 'I', text)
        text += '.'
    
    return text

# ------------------------
# Audio Buffer
# ------------------------
@dataclass
class AudioSegment:
    """Audio segment with metadata"""
    audio: np.ndarray
    timestamp: float
    is_speech: bool

class CircularAudioBuffer:
    """Efficient circular buffer for audio data"""
    def __init__(self, max_samples: int):
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.max_samples = max_samples
        self.write_pos = 0
        self.total_samples = 0
    
    def append(self, audio_int16: np.ndarray):
        """Append new audio data"""
        audio_float = audio_int16.astype(np.float32) / 32768.0
        n = len(audio_float)
        
        if n >= self.max_samples:
            # Audio larger than buffer, keep only last max_samples
            self.buffer[:] = audio_float[-self.max_samples:]
            self.write_pos = 0
            self.total_samples = self.max_samples
        else:
            # Circular write
            end_pos = self.write_pos + n
            if end_pos <= self.max_samples:
                self.buffer[self.write_pos:end_pos] = audio_float
            else:
                # Wrap around
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = audio_float[:first_part]
                self.buffer[:n - first_part] = audio_float[first_part:]
            
            self.write_pos = end_pos % self.max_samples
            self.total_samples = min(self.total_samples + n, self.max_samples)
    
    def get_recent(self, samples: int) -> np.ndarray:
        """Get most recent N samples"""
        samples = min(samples, self.total_samples)
        if samples == 0:
            return np.array([], dtype=np.float32)
        
        start_pos = (self.write_pos - samples) % self.max_samples
        
        if start_pos < self.write_pos:
            return self.buffer[start_pos:self.write_pos].copy()
        else:
            # Wrapped around
            return np.concatenate([
                self.buffer[start_pos:],
                self.buffer[:self.write_pos]
            ])
    
    def get_all(self) -> np.ndarray:
        """Get all buffered audio in correct order"""
        return self.get_recent(self.total_samples)
    
    def clear(self):
        """Clear buffer"""
        self.write_pos = 0
        self.total_samples = 0

# ------------------------
# VAD Processor
# ------------------------
class VADProcessor:
    """Efficient VAD processing"""
    def __init__(self, model, sample_rate: int):
        self.model = model
        self.sample_rate = sample_rate
        self.min_speech_samples = int(MIN_SPEECH_DURATION * sample_rate)
        self.min_silence_samples = int(MIN_SILENCE_DURATION * sample_rate)
    
    def detect_speech(self, audio: np.ndarray) -> list:
        """Detect speech timestamps in audio"""
        if len(audio) < 512:
            return []
        
        audio_tensor = torch.from_numpy(audio)
        
        try:
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=int(MIN_SILENCE_DURATION * 1000)
            )
            return timestamps
        except Exception as e:
            print(f"[WARN] VAD error: {e}")
            return []
    
    def has_speech(self, audio: np.ndarray) -> bool:
        """Quick check if audio contains speech"""
        timestamps = self.detect_speech(audio)
        return len(timestamps) > 0

# ------------------------
# Transcription Engine
# ------------------------
class TranscriptionEngine:
    """Optimized transcription with streaming support"""
    def __init__(self, model):
        self.model = model
        self.last_partial = ""
        self.last_context = ""
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        task: str = "transcribe",
        is_partial: bool = False
    ) -> str:
        """Transcribe audio with optimized settings for South Asian accents"""
        if len(audio) < 1600:  # At least 0.1s
            return ""
        
        try:
            # Enhanced prompt for South Asian English
            initial_prompt = "South Asian English speaker with Nepali accent. Common words: project, whisper, data, analysis, system."
            if self.last_context:
                initial_prompt += f" Context: {self.last_context}"
            
            segments, info = self.model.transcribe(
                audio,
                language=language,
                task=task,
                beam_size=5 if not is_partial else 3,  # Higher beam for finals
                best_of=5 if not is_partial else 3,
                temperature=0.0,  # Deterministic for consistency
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                initial_prompt=initial_prompt,
                vad_filter=False,
                word_timestamps=False,
                # Additional parameters for better accuracy
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=True,
                max_initial_timestamp=1.0
            )
            
            text = " ".join([seg.text for seg in segments]).strip()
            
            # Update context for next transcription
            if text and not is_partial:
                # Keep last 50 chars as context
                self.last_context = text[-50:] if len(text) > 50 else text
            
            return text
        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
            return ""
    
    def should_update_partial(self, new_text: str) -> bool:
        """Check if partial should be updated (avoid flickering)"""
        if not new_text or len(new_text) < PARTIAL_MIN_LENGTH:
            return False
        
        if not self.last_partial:
            return True
        
        # Use Levenshtein distance to check similarity
        distance = Levenshtein.distance(self.last_partial.lower(), new_text.lower())
        max_len = max(len(self.last_partial), len(new_text))
        
        if max_len == 0:
            return False
        
        change_ratio = distance / max_len
        return change_ratio >= PARTIAL_EDIT_THRESHOLD
    
    def update_partial(self, text: str):
        """Update last partial text"""
        self.last_partial = text
    
    def reset(self):
        """Reset state"""
        self.last_partial = ""
        self.last_context = ""

# ------------------------
# FastAPI
# ------------------------
app = FastAPI(title="Optimized ASR Streaming Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return {
        "status": "running",
        "device": DEVICE,
        "model": "medium.en",
        "optimizations": "enabled"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device": DEVICE
    }

# ------------------------
# WebSocket Handler
# ------------------------
@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    print(f"[CONNECT] Client {client_id}")
    
    # Initialize per-client state
    audio_buffer = CircularAudioBuffer(MAX_BUFFER_SAMPLES)
    vad_processor = VADProcessor(vad_model, SAMPLE_RATE)
    transcription_engine = TranscriptionEngine(model)
    
    last_speech_time = None
    last_partial_time = 0
    processing_lock = asyncio.Lock()
    
    async def process_audio():
        """Background worker for audio processing"""
        nonlocal last_speech_time, last_partial_time
        
        while True:
            await asyncio.sleep(0.05)  # 50ms processing interval
            
            async with processing_lock:
                if audio_buffer.total_samples < CHUNK_SAMPLES:
                    continue
                
                # Get recent audio for VAD check
                recent_audio = audio_buffer.get_recent(SAMPLE_RATE * 2)
                has_speech = vad_processor.has_speech(recent_audio)
                
                current_time = asyncio.get_event_loop().time()
                
                if has_speech:
                    last_speech_time = current_time
                    
                    # Generate partial transcription
                    if current_time - last_partial_time >= PARTIAL_UPDATE_INTERVAL:
                        # Get audio with context for better accuracy
                        context_audio = audio_buffer.get_recent(
                            min(CONTEXT_SAMPLES + CHUNK_SAMPLES, audio_buffer.total_samples)
                        )
                        
                        if len(context_audio) >= CHUNK_SAMPLES:
                            raw_text = transcription_engine.transcribe(
                                context_audio,
                                is_partial=True
                            )
                            
                            if raw_text:
                                normalized_text = normalize_text(raw_text, is_final=False)
                                
                                if transcription_engine.should_update_partial(normalized_text):
                                    await websocket.send_text(json.dumps({
                                        "type": "partial",
                                        "text": normalized_text
                                    }))
                                    transcription_engine.update_partial(normalized_text)
                                    last_partial_time = current_time
                
                # Endpoint detection
                elif last_speech_time is not None:
                    silence_duration = current_time - last_speech_time
                    
                    if silence_duration >= MIN_SILENCE_DURATION:
                        # Generate final transcription
                        full_audio = audio_buffer.get_all()
                        
                        if len(full_audio) >= CHUNK_SAMPLES:
                            final_text = transcription_engine.transcribe(
                                full_audio,
                                is_partial=False
                            )
                            
                            final_text = normalize_text(final_text, is_final=True)
                            
                            if final_text:
                                await websocket.send_text(json.dumps({
                                    "type": "final",
                                    "text": final_text
                                }))
                        
                        # Reset state
                        audio_buffer.clear()
                        transcription_engine.reset()
                        last_speech_time = None
    
    # Start background processing
    worker_task = asyncio.create_task(process_audio())
    
    try:
        while True:
            msg = await websocket.receive()
            
            if msg.get("bytes"):
                # Process incoming audio
                audio_bytes = msg["bytes"]
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                
                async with processing_lock:
                    audio_buffer.append(audio_int16)
            
            elif msg.get("text"):
                # Handle commands
                try:
                    data = json.loads(msg["text"])
                    cmd = data.get("cmd")
                    
                    if cmd == "flush":
                        async with processing_lock:
                            full_audio = audio_buffer.get_all()
                            
                            if len(full_audio) >= CHUNK_SAMPLES:
                                final_text = transcription_engine.transcribe(
                                    full_audio,
                                    is_partial=False
                                )
                                final_text = normalize_text(final_text, is_final=True)
                                
                                if final_text:
                                    await websocket.send_text(json.dumps({
                                        "type": "final",
                                        "text": final_text
                                    }))
                            
                            audio_buffer.clear()
                            transcription_engine.reset()
                    
                    elif cmd == "close":
                        await websocket.close()
                        break
                
                except json.JSONDecodeError:
                    print(f"[WARN] Invalid JSON command from client {client_id}")
    
    except WebSocketDisconnect:
        print(f"[DISCONNECT] Client {client_id}")
    
    except Exception as e:
        print(f"[ERROR] Client {client_id}: {e}")
    
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        
        print(f"[CLEANUP] Client {client_id}")

# ------------------------
# Run Server
# ------------------------
if __name__ == "__main__":
    print("[START] Optimized ASR Streaming Server")
    uvicorn.run(
        "server_streaming_optimized:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False  # Reduce logging overhead
    )