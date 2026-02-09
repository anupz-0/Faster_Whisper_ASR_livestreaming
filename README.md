Faster Whisper ASR – Live Streaming

A real-time speech-to-text server using Faster Whisper for fast and accurate transcription, optimized for South Asian English accents. Runs on GPU and can be streamed in real-time via WebSocket.

This repository includes:

Server_streaming_ultra.py – Fast, chunk-based ASR server using Faster Whisper + Silero VAD + SymSpell spelling correction.

requirements.txt – All Python dependencies.

frequency_dictionary_en_82_765.txt & frequency_bigramdictionary_en_243_342.txt – SymSpell dictionaries for improved transcription.

notebooks/Faster_Whisper_ASR_livestreaming.ipynb – Colab notebook for one-click cloud execution.

Features

✅ Real-time streaming ASR over WebSocket

✅ Supports South Asian English accents better than default Whisper models

✅ GPU acceleration (Faster Whisper)

✅ VAD-based endpointing for chunked transcription

✅ Spell correction & filler word removal

✅ Ultra-light Colab integration for 1-hour daily runs

Folder Structure
faster_asr_template/
├─ Server_streaming_ultra.py
├─ requirements.txt
├─ frequency_dictionary_en_82_765.txt
├─ frequency_bigramdictionary_en_243_342.txt
└─ notebooks/Faster_Whisper_ASR_livestreaming.ipynb

Setup – Google Colab (Recommended)

Open the notebook: Faster_Whisper_ASR_livestreaming.ipynb
 in Colab.

Set Runtime → Change runtime type → GPU.

Run the first cell – it will:

Install dependencies

Download SymSpell dictionaries

Start the ASR server

After this, the server runs and can accept WebSocket streams in real-time.

Setup – Local (Optional)

Clone the repo:

git clone https://github.com/YOUR_USERNAME/faster_asr_template.git
cd faster_asr_template


Install dependencies:

pip install -r requirements.txt


Start the ASR server:

python Server_streaming_ultra.py


Connect your client to ws://localhost:8000/ws/asr (or ngrok URL for public access).

Usage

WebSocket endpoint: /ws/asr

Audio format: PCM16LE, mono, 16 kHz

Control commands (JSON over WS):

{"cmd":"flush"} → force final transcription of current buffer

{"cmd":"close"} → close the connection

Server sends:

{"type":"partial","text":"..."} → partial stable transcript

{"type":"final","text":"..."} → final transcript after speech endpoint

Tips

For best accuracy with South Asian English accents, use Faster Whisper medium.en (GPU recommended).

Run in Colab GPU runtime for faster transcription; CPU is slower.

You can limit usage to 1 hour/day on Colab to avoid session timeout.

License

MIT License – free to use, modify, and distribute.

Acknowledgements

Faster Whisper
 – Fast Whisper implementation

Silero VAD
 – Voice activity detection

SymSpell
 – Spell correction
