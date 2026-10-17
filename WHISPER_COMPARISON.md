# STT Engine Comparison: whisper.cpp vs. faster-whisper (Jetson AGX Thor)

A technical evaluation of Speech-to-Text (STT) inference engines for the **D-Bot** real-time voice pipeline running on the **NVIDIA Jetson AGX Thor** (Blackwell SM 11.0 / ARM64).

---

## Executive Summary

For real-time speech-to-speech conversational robotics on Jetson AGX Thor, **`whisper.cpp` is the superior choice** over `faster-whisper`. It delivers lower latency (~230 ms for a 4.5-second utterance), requires only **~570 MB VRAM** (Q5_0 quantized), runs as an isolated C++ daemon without Python GIL contention, and is compiled natively for Thor's Blackwell SM 11.0 architecture with Flash Attention.

---

## Detailed Comparison

### 1. Blackwell Architecture (Compute Capability 11.0 / sm_110) Support

- **`whisper.cpp` (Selected)**:
  - Compiled natively on Jetson Thor directly against the CUDA 12 toolkit with `ARCHS = 1100`.
  - Generates native Blackwell machine code with full support for hardware Tensor Cores and Flash Attention (`--flash-attn`).
- **`faster-whisper` (CTranslate2)**:
  - Relies on pre-built CTranslate2 binary wheels, which historically target older architectures (up to `sm_89` / `sm_90`).
  - Running unsupported or generic PTX code on Blackwell leads to JIT compilation overhead on startup, cuDNN version conflicts on ARM64 (`aarch64`), or degraded throughput.

---

### 2. VRAM Footprint & Coexistence with 27B LLM

The Jetson AGX Thor shares unified LPDDR5X memory between the CPU, STT, LLM, and TTS models:

| Engine & Model | Quantization | VRAM Consumption | Memory Impact |
|---|---|---|---|
| **`whisper.cpp`** (`ggml-large-v3-turbo-q5_0.bin`) | **Q5_0** | **~573 MB** | **Negligible**; leaves maximal VRAM for Qwen 27B |
| **`faster-whisper`** (`large-v3-turbo`) | FP16 / Int8 | ~1.5 GB – 3.2 GB | Competes directly with Ollama's 27B KV cache |

`whisper.cpp` saves nearly **1 to 2.5 GB of VRAM**, allowing `qwen3.8:27b` to run smoothly with large context limits without triggering memory swapping or out-of-memory errors.

---

### 3. Process Isolation vs. Python GIL Contention

- **`whisper.cpp` Architecture**:
  - Operates as a standalone C++ service (`whisper-server`) on port 8080.
  - Receives audio requests via async HTTP (`/inference`).
  - Operates outside the Python runtime entirely: **zero Python GIL contention**, zero garbage collection pauses, and zero interference with the `aiohttp` WebSocket server or `PiperVoice` TTS synthesis.
- **`faster-whisper` Architecture**:
  - Runs inside Python through CTranslate2 bindings.
  - Requires managing threading or multiprocessing pools inside Python, potentially locking the GIL during batch transcription or tensor allocations.

---

### 4. Turn-Based Latency for Conversational Speech

In real-time robotics, the robot responds to short user utterances (1 to 5 seconds of speech):

- **Measured `whisper.cpp` performance on Jetson Thor**:
  - Model: `Whisper Large-v3-Turbo Q5_0`
  - Audio length: **4.5 seconds** (71,680 samples @ 16 kHz)
  - Processing time: **234.4 ms**
  - Real-time factor (RTF): **~0.052x** (nearly 20x faster than real time)
- **`faster-whisper` expected performance**:
  - Often excels at batched long-form audio transcription (podcasts/lectures), but incurs higher per-call setup overhead on single-utterance conversational turns (~300–450 ms).

---

## Comparison Matrix

| Feature / Criteria | `whisper.cpp` (Current) | `faster-whisper` (CTranslate2) |
|---|---|---|
| **Target GPU Compatibility** | **Native SM 11.0 (Blackwell)** | Often lacks SM 11.0 binaries |
| **Flash Attention** | **Built-in (`--flash-attn`)** | Requires external FlashAttention builds |
| **Quantization Format** | **GGML (Q5_0, Q8_0, etc.)** | CTranslate2 Int8 / FP16 |
| **VRAM Consumption** | **~570 MB** | ~1.5 GB – 3.2 GB |
| **Process Model** | Independent C++ HTTP daemon | In-process Python bindings |
| **Python GIL Impact** | **None (zero GIL overhead)** | Can cause lock contention |
| **Short-Utterance Latency** | **~200 – 260 ms** | ~300 – 450 ms |
| **Conversational Robotics Fit** | **Optimal** | Sub-optimal |

---

## Conclusion

For the **D-Bot** deployment on **Jetson AGX Thor**, `whisper.cpp` provides the best balance of ultra-low latency, native Blackwell GPU acceleration, minimal memory footprint, and architectural stability.
