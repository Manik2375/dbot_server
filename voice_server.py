#!/usr/bin/env python3
"""
High-Performance Voice Pipeline Server for Jetson AGX Thor
===========================================================
Robot: DROBOT Campus Guide (Jetson Orin Nano client)
Server: Jetson AGX Thor (Blackwell SM 110 GPU)
Pipeline:
  1. Audio Input: Low-Latency WebSockets (PCM 16kHz / WAV)
  2. STT: Whisper Large-v3-Turbo (CUDA accelerated on Thor)
  3. Knowledge RAG: Jaypee University (JUIT) Campus Engine (raw_research.txt)
  4. LLM: Qwen 3.8 via Ollama API with direct zero-thinking streaming
  5. TTS: Piper in-memory with 'hw_female' (en_US-hfc_female-medium)
"""

import asyncio
import io
import json
import logging
import os
import re
import time
import wave
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple

import aiohttp
from aiohttp import web
from piper import PiperVoice

from knowledge_engine import KnowledgeEngine, CAMPUS_CORE_SUMMARY

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("voice_server")

# Server Configurations
HOST = "0.0.0.0"
PORT = 8765
WHISPER_URL = "http://127.0.0.1:8080/inference"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3.8:latest"
PIPER_MODEL_PATH = "/home/thor/dbot_work/models/piper/en_US-hfc_female-medium.onnx"
PIPER_CONFIG_PATH = "/home/thor/dbot_work/models/piper/en_US-hfc_female-medium.onnx.json"

# Audio constants
SAMPLE_RATE_INPUT = 16000
SAMPLE_RATE_OUTPUT = 22050

# Common Whisper silence hallucinations to filter out
HALLUCINATIONS = {
    "", "you", "thank you", "thank you.", "thanks", "thanks for watching",
    "thanks for watching.", "subtitles by", "bye", "bye.", "so", "the end",
    "...", "um", "uh", "you you"
}

# Sentence delimiters (protect common honorifics and abbreviations)
SENTENCE_END_PATTERN = re.compile(
    r"(?<!\bProf)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bSh)(?<!\bShri)(?<!\bNo)(?<!\bDept)([.!?\n]+)\s*"
)

DROBOT_SYSTEM_PROMPT = (
    "You are D-Bot, the polite, friendly, and helpful campus guide robot at Jaypee University of Information Technology (JUIT), Waknaghat, Solan.\n\n"
    "Key Verified Campus Facts:\n"
    "- Vice Chancellor (VC): Prof. (Dr.) Rajendra Kumar Sharma.\n"
    "- Chancellor: Sh. Kavinder Gupta (Hon'ble Governor of Himachal Pradesh).\n"
    "- Pro-Chancellor & Chairman: Sh. Manoj Gaur (Executive Chairman, Jaypee Group).\n"
    "- Chief Minister (CM) of Himachal Pradesh: Sukhvinder Singh Sukhu (Chief Minister of Himachal Pradesh).\n"
    "- Registrar & Dean of Students: Brigadier RK Sharma, SM (Retd.).\n"
    "- Drobotics Lab / e-Yantra Lab: Located in ECL 5 (Department of Electronics and Communication Engineering).\n"
    "- Embedded Systems & Robotics Lab: Located in ECL 7.\n"
    "- Heads of Departments (HODs): CSE/IT is Prof. (Dr.) Vivek Sehgal; ECE is Prof. (Dr.) Rajiv Kumar; Biotech is Prof. (Dr.) Sudhir Kumar; Civil is Prof. (Dr.) Ashish Kumar; Management (JBSW) is Prof. (Dr.) Amit Srivastava.\n\n"
    "Strict Behavioral & Speaking Rules:\n"
    "- Tone: Always be polite, warm, and courteous.\n"
    "- Length: Keep your response very short and concise: exactly 1 short spoken sentence (maximum 2 very brief sentences).\n"
    "- Out of Bounds / Unknown Info: If the user asks about anything you do not know, or anything outside verified JUIT campus information, politely respond: 'I am sorry, I do not know that. Please ask at the campus help desk.'\n"
    "- Never use bullet points, markdown, asterisks, or unpronounceable symbols."
)


class WhisperSTTClient:
    """Fast async client for Whisper CUDA server"""
    def __init__(self, url: str = WHISPER_URL):
        self.url = url
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self.session

    async def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Sends audio to whisper-server and returns (transcript, duration_ms)"""
        t0 = time.time()
        # If raw PCM, wrap into WAV header
        if not audio_bytes.startswith(b"RIFF"):
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE_INPUT)
                wf.writeframes(audio_bytes)
            wav_data = wav_buf.getvalue()
        else:
            wav_data = audio_bytes

        session = await self.get_session()
        data = aiohttp.FormData()
        data.add_field("file", wav_data, filename="input.wav", content_type="audio/wav")

        try:
            async with session.post(self.url, data=data) as resp:
                if resp.status == 200:
                    res_json = await resp.json()
                    text = res_json.get("text", "").strip()
                    # Clean trailing whitespace/newlines
                    text = re.sub(r"\s+", " ", text).strip()
                    duration_ms = (time.time() - t0) * 1000.0
                    return text, duration_ms
                else:
                    err_txt = await resp.text()
                    logger.error(f"Whisper server error ({resp.status}): {err_txt}")
                    return "", 0.0
        except Exception as e:
            logger.error(f"Whisper connection error: {e}")
            return "", 0.0

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class PiperTTSEngine:
    """Pre-loaded in-memory Piper TTS engine for sub-100ms synthesis"""
    def __init__(self, model_path: str = PIPER_MODEL_PATH, config_path: str = PIPER_CONFIG_PATH):
        self.model_path = model_path
        self.config_path = config_path
        self.voice: Optional[PiperVoice] = None
        self._load_voice()

    def _load_voice(self):
        t0 = time.time()
        logger.info(f"Loading Piper voice '{self.model_path}' into memory...")
        self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)
        logger.info(f"Piper voice loaded in {time.time() - t0:.3f}s")

    def synthesize_sentence(self, text: str) -> bytes:
        """Synthesizes text into raw PCM 16-bit 22050Hz audio bytes"""
        if not self.voice or not text.strip():
            return b""
        pcm_chunks = []
        for chunk in self.voice.synthesize(text):
            pcm_chunks.append(chunk.audio_int16_bytes)
        return b"".join(pcm_chunks)

    def synthesize_sentence_wav(self, text: str) -> bytes:
        """Synthesizes text and wraps in valid WAV container"""
        pcm_bytes = self.synthesize_sentence(text)
        if not pcm_bytes:
            return b""
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_OUTPUT)
            wf.writeframes(pcm_bytes)
        return wav_buf.getvalue()


class VoiceServer:
    def __init__(self):
        self.stt = WhisperSTTClient()
        self.tts = PiperTTSEngine()
        self.rag = KnowledgeEngine()
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info("Voice Server initialized with all engines ready.")

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self.session

    async def stream_llm(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """
        Streams Qwen 3.8 tokens with prefilled `<think>\n</think>` to eliminate thinking latency.
        Yields clean, incremental text tokens.
        """
        prompt = (
            f"<|im_start|>system\n{DROBOT_SYSTEM_PROMPT}\n\nCampus Facts:\n{context}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "keep_alive": -1,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 120
            }
        }

        session = await self.get_session()
        async with session.post(OLLAMA_URL, json=payload) as resp:
            if resp.status != 200:
                err = await resp.text()
                logger.error(f"Ollama error {resp.status}: {err}")
                yield "I apologize, but I am currently having trouble processing that query."
                return

            async for line in resp.content:
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        # Filter out potential internal tokens
                        if "<think>" in token or "</think>" in token or "<|im_end|>" in token:
                            continue
                        if token:
                            yield token
                    except Exception:
                        continue

    async def process_utterance(self, ws: web.WebSocketResponse, audio_data: bytes):
        """Processes an audio utterance through the STT -> RAG -> LLM -> TTS pipeline"""
        total_t0 = time.time()
        logger.info(f"Incoming audio utterance: {len(audio_data)} bytes")

        # 1. STT via Whisper CUDA
        await ws.send_json({"type": "status", "stage": "stt", "message": "Transcribing speech on GPU..."})
        transcript, stt_ms = await self.stt.transcribe(audio_data)

        # Anti-hallucination check
        cleaned_transcript = transcript.lower().strip()
        if not cleaned_transcript or cleaned_transcript in HALLUCINATIONS or len(cleaned_transcript) <= 2:
            logger.info(f"Ignoring silent/hallucinated transcript: '{transcript}'")
            await ws.send_json({"type": "stt_result", "transcript": "", "ignored": True})
            return

        logger.info(f"✓ STT Result ({stt_ms:.1f}ms): '{transcript}'")
        await ws.send_json({
            "type": "stt_result",
            "transcript": transcript,
            "latency_ms": round(stt_ms, 1)
        })

        # 2. Knowledge RAG
        rag_t0 = time.time()
        context = self.rag.format_context(transcript)
        rag_ms = (time.time() - rag_t0) * 1000.0
        logger.info(f"✓ RAG Context retrieved in {rag_ms:.1f}ms")

        # 3. LLM Generation + 4. Streaming Sentence TTS
        await ws.send_json({"type": "status", "stage": "llm", "message": "D-Bot thinking..."})
        
        token_buffer = ""
        full_response = []
        sentence_index = 0
        first_token_time = None


        async for token in self.stream_llm(transcript, context):
            if first_token_time is None:
                first_token_time = time.time()
                ttft_ms = (first_token_time - total_t0) * 1000.0
                await ws.send_json({"type": "llm_ttft", "ttft_ms": round(ttft_ms, 1)})

            token_buffer += token
            full_response.append(token)
            await ws.send_json({"type": "llm_chunk", "token": token})

            # Check if a sentence is complete
            match = SENTENCE_END_PATTERN.search(token_buffer)
            if match:
                end_pos = match.end()
                sentence = token_buffer[:end_pos].strip()
                token_buffer = token_buffer[end_pos:]

                if len(sentence) > 3:
                    await self._synthesize_and_send(ws, sentence, sentence_index, total_t0)
                    sentence_index += 1

        # Process any leftover text
        remaining = token_buffer.strip()
        if remaining and len(remaining) > 2:
            await self._synthesize_and_send(ws, remaining, sentence_index, total_t0)
            sentence_index += 1

        total_ms = (time.time() - total_t0) * 1000.0
        full_text = "".join(full_response).strip()
        logger.info(f"✓ Turn complete in {total_ms:.1f}ms. Total sentences: {sentence_index}")

        await ws.send_json({
            "type": "turn_complete",
            "full_text": full_text,
            "total_latency_ms": round(total_ms, 1)
        })

    async def _synthesize_and_send(self, ws: web.WebSocketResponse, sentence: str, index: int, pipeline_start_time: float):
        """Synthesizes a single sentence and sends audio chunk over WebSocket"""
        tts_t0 = time.time()
        # Clean sentence from any symbols that shouldn't be read out
        clean_text = re.sub(r"[\*\#\_\[\]]", "", sentence).strip()
        if not clean_text:
            return

        # Run CPU ONNX synthesis in worker thread so event loop remains unblocked
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, self.tts.synthesize_sentence_wav, clean_text)

        tts_ms = (time.time() - tts_t0) * 1000.0
        ttfa_ms = (time.time() - pipeline_start_time) * 1000.0

        logger.info(f"  Sentence {index} TTS ({tts_ms:.1f}ms, TTFA {ttfa_ms:.1f}ms): '{clean_text}'")

        # 1. Header JSON
        await ws.send_json({
            "type": "audio_start",
            "sentence_index": index,
            "sentence": clean_text,
            "sample_rate": SAMPLE_RATE_OUTPUT,
            "format": "wav",
            "tts_latency_ms": round(tts_ms, 1),
            "ttfa_ms": round(ttfa_ms, 1)
        })

        # 2. Binary audio frame
        await ws.send_bytes(wav_bytes)

        # 3. Footer JSON
        await ws.send_json({
            "type": "audio_end",
            "sentence_index": index
        })

    # ================= HTTP & WebSocket Route Handlers =================

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Handles incoming WebSocket connections from Orin Nano or Web clients"""
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        client_ip = request.remote
        logger.info(f"Client connected from {client_ip}")

        # Send welcome message
        await ws.send_json({
            "type": "session_ready",
            "version": "1.0",
            "robot": "D-Bot",
            "piper_voice": "en_US-hfc_female-medium",
            "whisper_model": "large-v3-turbo (CUDA)",
            "llm_model": OLLAMA_MODEL
        })

        audio_buffer = bytearray()
        is_streaming_audio = False

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary frame received
                    data = msg.data
                    if is_streaming_audio:
                        audio_buffer.extend(data)
                    else:
                        # Full standalone audio utterance
                        asyncio.create_task(self.process_utterance(ws, data))

                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                        msg_type = payload.get("type", "")

                        if msg_type == "audio_start":
                            audio_buffer.clear()
                            is_streaming_audio = True

                        elif msg_type == "audio_chunk":
                            import base64
                            b64 = payload.get("data", "")
                            audio_buffer.extend(base64.b64decode(b64))

                        elif msg_type == "audio_end":
                            is_streaming_audio = False
                            if audio_buffer:
                                audio_to_process = bytes(audio_buffer)
                                audio_buffer.clear()
                                asyncio.create_task(self.process_utterance(ws, audio_to_process))

                        elif msg_type == "text_query":
                            # Direct text test query
                            query = payload.get("text", "")
                            if query:
                                context = self.rag.format_context(query)
                                sentence_index = 0
                                token_buf = ""
                                t0 = time.time()
                                full_resp = []

                                async for token in self.stream_llm(query, context):
                                    token_buf += token
                                    full_resp.append(token)
                                    await ws.send_json({"type": "llm_chunk", "token": token})

                                    match = SENTENCE_END_PATTERN.search(token_buf)
                                    if match:
                                        end_pos = match.end()
                                        s = token_buf[:end_pos].strip()
                                        token_buf = token_buf[end_pos:]
                                        if len(s) > 3:
                                            await self._synthesize_and_send(ws, s, sentence_index, t0)
                                            sentence_index += 1

                                remaining_text = token_buf.strip()
                                if remaining_text and len(remaining_text) > 2:
                                    await self._synthesize_and_send(ws, remaining_text, sentence_index, t0)
                                    sentence_index += 1

                                await ws.send_json({
                                    "type": "turn_complete",
                                    "full_text": "".join(full_resp).strip(),
                                    "total_latency_ms": round((time.time() - t0) * 1000.0, 1)
                                })

                        elif msg_type == "ping":
                            await ws.send_json({"type": "pong", "time": time.time()})

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {msg.data[:100]}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WS error: {ws.exception()}")

        finally:
            logger.info(f"Client disconnected: {client_ip}")

        return ws

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint for system monitoring"""
        whisper_ok = False
        try:
            session = await self.get_session()
            async with session.get(f"http://127.0.0.1:8080/health", timeout=1.0) as resp:
                whisper_ok = (resp.status == 200)
        except Exception:
            # Check port open
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            whisper_ok = (sock.connect_ex(('127.0.0.1', 8080)) == 0)
            sock.close()

        ollama_ok = False
        try:
            session = await self.get_session()
            async with session.get("http://127.0.0.1:11434/api/tags", timeout=1.0) as resp:
                ollama_ok = (resp.status == 200)
        except Exception:
            pass

        return web.json_response({
            "status": "healthy" if (whisper_ok and ollama_ok) else "degraded",
            "engines": {
                "whisper_cuda": whisper_ok,
                "ollama_qwen38": ollama_ok,
                "piper_tts": self.tts.voice is not None,
                "knowledge_engine": len(self.rag.chunks) > 0
            },
            "piper_voice": "en_US-hfc_female-medium (hw_female)",
            "robot": "D-Bot",
            "host": HOST,
            "port": PORT
        })

    async def handle_index(self, request: web.Request) -> web.Response:
        """Interactive Web UI for live testing from browser"""
        html_path = "/home/thor/dbot_work/web_ui.html"
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return web.Response(text=f.read(), content_type="text/html")
        return web.Response(text="D-Bot Voice Server is active. Connect via WebSocket to /ws", content_type="text/plain")


async def init_app():
    server = VoiceServer()
    app = web.Application()
    app.router.add_get("/", server.handle_index)
    app.router.add_get("/health", server.handle_health)
    app.router.add_get("/ws", server.handle_ws)
    return app


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🤖 Starting D-Bot Voice Server on Jetson AGX Thor...")
    logger.info(f"   Listening on: http://{HOST}:{PORT}")
    logger.info(f"   WebSocket URL: ws://{HOST}:{PORT}/ws")
    logger.info("=" * 70)
    web.run_app(init_app(), host=HOST, port=PORT)
