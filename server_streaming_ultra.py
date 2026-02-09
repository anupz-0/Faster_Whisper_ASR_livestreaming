from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json, os
import numpy as np
import torch
import uvicorn
from collections import deque

from symspellpy.symspellpy import SymSpell
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel

# ------------------------
# Configuration
# ------------------------
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
OVERLAP_SECONDS = 0.3                 # smaller overlap = faster partials
OVERLAP_SAMPLES = int(SAMPLE_RATE * OVERLAP_SECONDS)
MAX_BUFFER_SECONDS = 60               # ultra long audio support
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)

ENDPOINT_SILENCE_SECONDS = 0.6        # faster endpoint detection
STABILITY_WINDOW = 1                  # ultra-stable partials (1 repeat enough)
PARTIAL_MIN_CHARS = 5                 # don't send tiny partials
PARTIAL_COOLDOWN = 0.2                # throttle partial sends

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

# ------------------------
# Faster Whisper Model
# ------------------------
print("[INFO] Loading Faster-Whisper medium.en...")
model = WhisperModel(
    "medium.en",
    device=DEVICE,
    compute_type="float16" if DEVICE=="cuda" else "int8"
)

# ------------------------
# Load Silero VAD
# ------------------------
vad_model = load_silero_vad()

# ------------------------
# SymSpell Setup
# ------------------------
SYM_MAX_EDIT_DIST = 2
SYM_PREFIX_LEN = 7
BASE_DIR = os.path.dirname(__file__)

sym_spell = SymSpell(max_dictionary_edit_distance=SYM_MAX_EDIT_DIST, prefix_length=SYM_PREFIX_LEN)
sym_spell.load_dictionary(os.path.join(BASE_DIR, "frequency_dictionary_en_82_765.txt"),0,1)
sym_spell.load_bigram_dictionary(os.path.join(BASE_DIR, "frequency_bigramdictionary_en_243_342.txt"),0,2)

def symspell_correct_text(text):
    try:
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=SYM_MAX_EDIT_DIST)
        return suggestions[0].term if suggestions else text
    except:
        return text

FILLERS = {"um","umm","uh","uhh","uhm","erm","hmm","ah","aah","aa","eh"}
def remove_fillers(text):
    return " ".join([w for w in text.split() if w.lower() not in FILLERS])

# ------------------------
# FastAPI
# ------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return {"status":"running"}

# ------------------------
# Transcription Function
# ------------------------
def transcribe_audio_float32(audio_float32):
    try:
        segments, _ = model.transcribe(
            audio_float32,
            language="en",
            beam_size=5,
            vad_filter=False,
            initial_prompt="South Asian English speaker, Nepali accent"
        )
        return " ".join([seg.text for seg in segments]).strip()
    except:
        return ""

# ------------------------
# WebSocket ASR Ultra
# ------------------------
@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Client connected")

    audio_buffer_bytes = bytearray()
    last_speech_time = None
    partial_text_cache = ""
    last_partial_send = 0

    async def transcribe_worker():
        nonlocal audio_buffer_bytes,last_speech_time,partial_text_cache,last_partial_send
        while True:
            await asyncio.sleep(0.1)
            if len(audio_buffer_bytes)<320:
                continue

            # VAD check
            vad_bytes = audio_buffer_bytes[-SAMPLE_RATE*2:]
            int16_vad = np.frombuffer(vad_bytes,dtype=np.int16)
            audio_tensor = torch.from_numpy(int16_vad.astype(np.float32)/32768.0)
            try:
                speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=SAMPLE_RATE)
            except:
                speech_timestamps = []

            speech_present = len(speech_timestamps)>0
            now = asyncio.get_event_loop().time()

            if speech_present:
                last_speech_time = now
                start = max(0, len(audio_buffer_bytes)-(CHUNK_SAMPLES+OVERLAP_SAMPLES)*2)
                proc_bytes = audio_buffer_bytes[start:]
                audio_float = np.frombuffer(proc_bytes,dtype=np.int16).astype(np.float32)/32768.0

                raw_text = transcribe_audio_float32(audio_float)
                if raw_text and len(raw_text)>=PARTIAL_MIN_CHARS:
                    text = symspell_correct_text(raw_text)
                    text = remove_fillers(text)

                    # throttle partial sends
                    if text != partial_text_cache and now-last_partial_send>PARTIAL_COOLDOWN:
                        await websocket.send_text(json.dumps({"type":"partial","text":text}))
                        partial_text_cache=text
                        last_partial_send=now

            # Endpoint detection
            if last_speech_time and now-last_speech_time>ENDPOINT_SILENCE_SECONDS:
                if len(audio_buffer_bytes)>=320:
                    audio_float_all = np.frombuffer(audio_buffer_bytes,dtype=np.int16).astype(np.float32)/32768.0
                    final_text = transcribe_audio_float32(audio_float_all)
                    final_text = remove_fillers(symspell_correct_text(final_text))
                    if final_text:
                        await websocket.send_text(json.dumps({"type":"final","text":final_text}))

                audio_buffer_bytes=bytearray()
                partial_text_cache=""
                last_speech_time=None

    worker_task = asyncio.create_task(transcribe_worker())

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes"):
                audio_buffer_bytes.extend(msg["bytes"])
                if len(audio_buffer_bytes) > MAX_BUFFER_SAMPLES*2:
                    audio_buffer_bytes = audio_buffer_bytes[-MAX_BUFFER_SAMPLES*2:]
            elif msg.get("text"):
                j=json.loads(msg["text"])
                cmd=j.get("cmd")
                if cmd=="flush":
                    if len(audio_buffer_bytes)>=320:
                        audio_float_all = np.frombuffer(audio_buffer_bytes,dtype=np.int16).astype(np.float32)/32768.0
                        final_text = remove_fillers(symspell_correct_text(transcribe_audio_float32(audio_float_all)))
                        if final_text:
                            await websocket.send_text(json.dumps({"type":"final","text":final_text}))
                    audio_buffer_bytes=bytearray()
                elif cmd=="close":
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        print("[INFO] Client disconnected")
    finally:
        worker_task.cancel()

# ------------------------
# Run server
# ------------------------
def run_server():
    import uvicorn
    uvicorn.run("server_streaming_ultra:app", host="0.0.0.0", port=8000)

