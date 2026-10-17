# D-Bot Voice Pipeline Server (Jetson AGX Thor)

High-performance, low-latency Speech-to-Speech (S2S) voice server designed for the **D-Bot** campus guide robot at Jaypee University of Information Technology (JUIT), Waknaghat, Solan.

---

## System Architecture

```
[Jetson Orin Nano (Robot D-Bot)]
       │
       │ WebSocket (ws://<JETSON_THOR_IP>:8765/ws)
       ▼
[Jetson AGX Thor]
  ├─► 1. STT: Whisper Large-v3-Turbo Q5_0 (CUDA SM 11.0 Blackwell GPU, port 8080, ~230ms)
  ├─► 2. RAG: Campus Knowledge Engine (BM25 + keyword retrieval, <1ms)
  ├─► 3. LLM: Qwen 3.8 / 27B (Ollama GPU, zero-thinking streaming TTFT ~580ms)
  └─► 4. TTS: Piper in-memory ('en_US-hfc_female-medium' ONNX, ~80ms/sentence)
       │
       ▼ Streamed Sentence Audio Chunks (16-bit 22050Hz WAV/PCM)
[Jetson Orin Nano Speaker Playback]
```

> [!TIP]
> **STT Engine Selection (`whisper.cpp` vs. `faster-whisper`)**: For a detailed technical comparison of why `whisper.cpp` was selected over `faster-whisper` on the Jetson AGX Thor (native SM 11.0 Blackwell architecture, ~570 MB VRAM footprint, and ~230 ms latency), see [WHISPER_COMPARISON.md](WHISPER_COMPARISON.md).

---

## Robot Persona & Behavioral Boundaries

D-Bot is programmed with a clear persona and guardrails designed for real-time robotic interaction:

- **Identity**: Autonomous, friendly campus guide robot on wheels rolling around the JUIT Waknaghat campus.
- **Concise Spoken Responses**: Answers are strictly kept to **1–2 spoken sentences** to prevent the robot from rambling or holding long speeches in public corridors.
- **Large/Complex Answers**: If an explanation is very broad or complex, D-Bot delivers a brief 1-sentence summary and directs the user to Google or check online for full details.
- **Code Request Boundary**: If asked to write code (C++, Python, Java, etc.), D-Bot politely declines, explains that it is a physical robot on wheels rather than a coding assistant, and advises writing and testing code on a computer or in the campus computer labs.
- **Humor & Decorum**: Only tells clean, wholesome, family-friendly jokes (no bad/inappropriate jokes). Never mocks, teases, belittles, or makes fun of any person, topic, or situation.
- **Open & Helpful**: Answers general questions, short stories, clean jokes, and math queries without rigid refusal, while grounding all campus questions in verified JUIT facts.

---

## Jetson Thor Network Endpoints

- **Wi-Fi AP IP**: `10.42.0.1` *(Default access point for robot)*
- **Ethernet IP**: `172.16.32.29`
- **USB Device IP**: `192.168.55.1`

### Service URLs:
- **WebSocket Voice Pipeline**: `ws://10.42.0.1:8765/ws` (or `ws://172.16.32.29:8765/ws`)
- **Web Diagnostic Console**: `http://10.42.0.1:8765/` (interactive dark-mode web tester)
- **Health Check Endpoint**: `curl http://localhost:8765/health`

---

## Installation & Setup

### 1. Prerequisites
- NVIDIA Jetson AGX Thor (JetPack / CUDA 12.x / SM 11.0 Blackwell) or Linux GPU host.
- Python 3.10+ (tested on Python 3.12).
- [Ollama](https://ollama.com/) installed and running.

### 2. Python Environment Setup
```bash
cd /home/thor/dbot_work

# Create and activate virtual environment
python3 -m venv voice_agent
source voice_agent/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Weights Setup
Ensure the model files are placed in their respective directories:

```
models/
├── piper/
│   ├── en_US-hfc_female-medium.onnx
│   └── en_US-hfc_female-medium.onnx.json
└── whisper/
    └── ggml-large-v3-turbo-q5_0.bin
```

### 4. Ollama LLM Setup
Ensure the Qwen 3.8 model is pulled and aliased:
```bash
# Start Ollama service (if not running as a systemd service)
ollama serve &

# Pull model and ensure alias
ollama pull qwen3.8:latest
ollama cp qwen3.8:latest qwen3.8:27b
```

---

## Running the Server

### Master Launcher (Recommended)
The master launcher automatically checks/starts Ollama, spawns the CUDA Whisper server on port 8080, and runs `voice_server.py`:

```bash
# Normal start
./run_server.sh

# Debug mode (streams real-time background Whisper logs with [WHISPER] prefix)
./run_server.sh --debug
```

### Manual Individual Service Launch (Optional)
```bash
# 1. Start Whisper CUDA server (port 8080)
./start_whisper_server.sh

# 2. Start Main Voice WebSocket Server (port 8765)
./voice_agent/bin/python voice_server.py
```

---

## Client Usage & Testing

### 1. Local Command-Line Testing (`drobot_client.py`)
```bash
# Test with text input
./voice_agent/bin/python drobot_client.py --text "Where is the Drobotics lab located?"

# Test with a question requiring brevity
./voice_agent/bin/python drobot_client.py --text "Explain how quantum computers work"

# Test code boundary rejection
./voice_agent/bin/python drobot_client.py --text "Write me C++ code for binary search"

# Test with a prerecorded WAV file
./voice_agent/bin/python drobot_client.py --test-file hello_test.wav

# Run latency benchmark test
./voice_agent/bin/python drobot_client.py --benchmark
```

### 2. Web GUI Test Console
Open `http://<SERVER_IP>:8765/` in any browser on the network to access the web console:
- Live microphone recording and playback.
- Text query input with real-time streaming token display.
- Latency meters for STT, RAG, LLM TTFT, and TTS.

### 3. Running on Jetson Orin Nano (Robot Client)
Copy `drobot_client.py` to the Jetson Orin Nano on D-Bot:
```bash
pip install websockets

# Interactive voice mode: records from robot mic, streams to Thor, plays response on robot speakers
python3 drobot_client.py --server ws://10.42.0.1:8765/ws --interactive
```

---

## Performance Benchmarks (Jetson AGX Thor)

| Pipeline Stage | Engine / Model | Latency |
|---|---|---|
| **Speech-to-Text (STT)** | Whisper Large-v3-Turbo Q5_0 (CUDA SM 11.0, Flash-Attn) | **200 – 260 ms** |
| **Knowledge Retrieval (RAG)** | Hybrid BM25 & Keyword Retrieval (`knowledge_engine.py`) | **< 1 ms** |
| **LLM Time-To-First-Token (TTFT)** | Qwen 3.8 / 27B (Ollama GPU, zero-thinking prefill) | **~350 – 550 ms** |
| **Text-to-Speech (TTS)** | Piper Neural TTS (`en_US-hfc_female-medium`, in-memory) | **~70 – 110 ms / sentence** |
| **Total Time-to-First-Audio (TTFA)** | Full end-to-end user voice input to robot speech | **~750 – 950 ms** |

---

## Repository Structure

```
├── run_server.sh              # Master launcher script for all services
├── start_whisper_server.sh    # Launches Whisper.cpp CUDA daemon (port 8080)
├── voice_server.py            # Main async WebSocket & HTTP server (port 8765)
├── knowledge_engine.py        # Campus Knowledge Engine & dynamic RAG retriever
├── drobot_client.py           # CLI test client & Jetson Orin Nano robot client
├── web_ui.html                # Interactive browser test bench & diagnostic console
├── requirements.txt           # Python dependencies for the voice pipeline
├── WHISPER_COMPARISON.md      # Technical comparison: whisper.cpp vs. faster-whisper on Thor
├── .gitignore                 # Excludes models, venv, caches, logs, and build files
├── clean_research.txt         # Clean structured facts about JUIT Waknaghat
├── raw_research.txt           # Raw reference research documentation
└── models/                    # Neural weights directory (Whisper & Piper)
```
