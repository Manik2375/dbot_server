# D-Bot Voice Pipeline Server (Jetson AGX Thor)

High-performance, low-latency Speech-to-Speech (S2S) voice server designed for the **D-Bot** campus guide robot at Jaypee University of Information Technology (JUIT), Waknaghat.

---

## System Architecture

```
[Jetson Orin Nano (Robot D-Bot)]
       │
       │ WebSocket (ws://<JETSON_THOR_IP>:8765/ws)
       ▼
[Jetson AGX Thor]
  ├─► 1. STT: Whisper Large-v3-Turbo (CUDA SM 11.0 Blackwell GPU, ~230ms)
  ├─► 2. RAG: Campus Knowledge Engine (raw_research.txt, <1ms)
  ├─► 3. LLM: Qwen 3.8 (Ollama GPU, zero-thinking streaming TTFT ~580ms)
  └─► 4. TTS: Piper in-memory ('hw_female' / en_US-hfc_female-medium, ~80ms/sentence)
       │
       ▼ Streamed Sentence Audio Chunks (WAV/PCM)
[Jetson Orin Nano Speaker Playback]
```

---

## Jetson Thor Network Endpoints

- **Ethernet IP**: `172.16.32.29`
- **Wi-Fi AP IP**: `10.42.0.1`
- **USB Device IP**: `192.168.55.1`

### WebSocket URL:
```
ws://172.16.32.29:8765/ws
```

### Web Testing Interface:
Open in browser from any machine on the network:
```
http://172.16.32.29:8765/
```

### Health Check Endpoint:
```bash
curl http://localhost:8765/health
```

---

## Quick Start Commands on Jetson Thor

### 1. Start Complete System (Whisper CUDA + Voice Server)
```bash
cd /home/thor/dbot_work
./run_server.sh
```

### 2. Run Local Diagnostic / Client Test
```bash
# Test with a question
python3 drobot_client.py --text "Where is the Drobotics lab located?"

# Test with an audio WAV file
python3 drobot_client.py --test-file test_tts.wav

# Interactive microphone test (if mic connected to Thor)
python3 drobot_client.py --interactive
```

---

## Running on Jetson Orin Nano (D-Bot Client)

Copy [`drobot_client.py`](file:///home/thor/dbot_work/drobot_client.py) to your Jetson Orin Nano:

```bash
# On Jetson Orin Nano
pip install websockets

# Interactive voice mode (records from robot mic and plays via robot speakers)
python3 drobot_client.py --server ws://172.16.32.29:8765/ws --interactive
```

---

## Performance Benchmark on Jetson Thor

| Stage | Technology | Latency |
|---|---|---|
| **STT** | Whisper Large-v3-Turbo (CUDA sm_110) | **230 – 290 ms** |
| **RAG** | Fast Hybrid BM25 / Keyword Retrieval | **< 1 ms** |
| **LLM TTFT** | Qwen 3.8 (Ollama GPU streaming) | **~580 ms** |
| **TTS** | Piper `en_US-hfc_female-medium` (In-Memory) | **~80 – 120 ms / sentence** |
| **Time-to-First-Audio (TTFA)** | Full Speech-to-Speech pipeline | **~1.7 – 2.5 s** |

---

## Files in this Directory

- [`voice_server.py`](file:///home/thor/dbot_work/voice_server.py): Main async WebSocket server handling STT, RAG, LLM streaming, and Piper TTS.
- [`knowledge_engine.py`](file:///home/thor/dbot_work/knowledge_engine.py): Campus facts retriever built from `raw_research.txt`.
- [`start_whisper_server.sh`](file:///home/thor/dbot_work/start_whisper_server.sh): Launches Whisper CUDA server daemon on port 8080.
- [`run_server.sh`](file:///home/thor/dbot_work/run_server.sh): Master start script for all services.
- [`drobot_client.py`](file:///home/thor/dbot_work/drobot_client.py): Reference WebSocket client for Jetson Orin Nano.
- [`web_ui.html`](file:///home/thor/dbot_work/web_ui.html): Dark-mode browser test console with audio recorder and latency gauges.
- `models/`: Pre-downloaded and linked Whisper model and Piper `en_US-hfc_female-medium` ONNX voice.
