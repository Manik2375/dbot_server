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

import argparse
import asyncio
import io
import json
import logging
import os
import re
import struct
import time
import wave
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple, Set

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
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8:27b")
PIPER_MODEL_PATH = "/home/thor/dbot_work/models/piper/en_US-hfc_female-medium.onnx"
PIPER_CONFIG_PATH = "/home/thor/dbot_work/models/piper/en_US-hfc_female-medium.onnx.json"
TOOLS_FILE_PATH = os.path.join(os.path.dirname(__file__), "tools.json")

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


def load_tools_definitions(file_path: str = TOOLS_FILE_PATH) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Loads Ollama tools schema from JSON file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                tools_data = json.load(f)
                allowed = {
                    t.get("function", {}).get("name")
                    for t in tools_data
                    if t.get("function", {}).get("name")
                }
                logger.info(f"Loaded {len(tools_data)} robot tools from {file_path}: {list(allowed)}")
                return tools_data, allowed
        else:
            logger.warning(f"Tools file not found at {file_path}")
    except Exception as e:
        logger.error(f"Error loading tools definition from {file_path}: {e}")
    return [], set()


DROBOT_SYSTEM_PROMPT = (
    "You are D-Bot, a friendly, cheerful, and helpful robot on wheels at Jaypee University of Information Technology (JUIT), Waknaghat, Solan. "
    "You roll around campus on wheels assisting and talking with students, faculty, and visitors.\n\n"
    "Key Verified Campus Facts:\n"
    "- Vice Chancellor (VC): Prof. (Dr.) Rajendra Kumar Sharma.\n"
    "- Chancellor: Sh. Kavinder Gupta (Hon'ble Governor of Himachal Pradesh).\n"
    "- Pro-Chancellor & Chairman: Sh. Manoj Gaur (Executive Chairman, Jaypee Group).\n"
    "- Chief Minister (CM) of Himachal Pradesh: Sukhvinder Singh Sukhu (Chief Minister of Himachal Pradesh).\n"
    "- Registrar & Dean of Students: Brigadier RK Sharma, SM (Retd.).\n"
    "- Drobotics Lab / e-Yantra Lab: Located in ECL 5 (Department of Electronics and Communication Engineering).\n"
    "- Embedded Systems & Robotics Lab: Located in ECL 7.\n"
    "- Heads of Departments (HODs): CSE/IT is Prof. (Dr.) Vivek Sehgal; ECE is Prof. (Dr.) Rajiv Kumar; Biotech is Prof. (Dr.) Sudhir Kumar; Civil is Prof. (Dr.) Ashish Kumar; Management (JBSW) is Prof. (Dr.) Amit Srivastava.\n\n"
    "Robot Physical Capabilities & Tool Calling:\n"
    "- You have physical motorized wheels and navigation systems.\n"
    "- When the user asks you to move, back up, rotate, turn, navigate somewhere on campus, deliver an item, or stop, you MUST call the appropriate tool.\n"
    "- Only call tools that are defined in your tool list (navigate_to, navigate_and_deliver, move_distance, rotate, stop). Never invent new tool names.\n"
    "- When calling a tool, provide a brief, clear spoken confirmation in your response content (e.g., 'Moving forward 1 meter now.').\n"
    "- If the user asks a normal question, joke, story, math problem, or campus information, do NOT call any tool; simply respond directly with spoken text in 1 to 2 sentences.\n\n"
    "Strict Rules & Boundaries:\n"
    "1. No Code Requests: If anyone asks you to write code, programs, or scripts in any programming language (like C++, Python, Java, etc.), politely reject the request. Explain that you are a physical robot on wheels, not a coding assistant, and suggest they write and test code on their computer or in the campus computer labs.\n"
    "2. Short Answers Only: Never keep talking or give lengthy lectures. Keep all spoken responses very short (maximum 1 to 2 spoken sentences). If an explanation is very big, complex, or long, give a quick one-sentence summary and tell them to Google it or check online for details.\n"
    "3. No Bad Jokes: If asked for a joke, tell only clean, wholesome, clever, and family-friendly jokes.\n"
    "4. Never Make Fun of Anything: Never mock, tease, belittle, or make fun of any person, topic, question, or situation. Always remain respectful, kind, and safe.\n"
    "5. Spoken Style: Keep speech natural and conversational. Never use bullet points, asterisks, markdown formatting, emojis, code blocks, or unpronounceable characters.\n"
    "6. No Internal Thinking: Do not use internal thinking, reasoning tags, or explanations. Immediately output the spoken response or tool call."
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
        self.tools, self.allowed_tool_names = load_tools_definitions()
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Voice Server initialized with {len(self.tools)} tools loaded: {list(self.allowed_tool_names)}")

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45))
        return self.session

    async def chat_with_tools(self, query: str, context: str) -> Dict[str, Any]:
        """
        Calls Ollama's /api/chat endpoint with tool calling enabled.
        Passes retrieved context and the user query.
        """
        system_content = DROBOT_SYSTEM_PROMPT
        if context.strip():
            system_content += f"\n\nCampus Facts:\n{context}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": self.tools,
            "stream": False,
            "think": False,
            "options": {
                "think": False,
                "temperature": 0.5,
                "top_p": 0.9,
                "num_predict": 160
            }
        }

        session = await self.get_session()
        try:
            async with session.post(OLLAMA_CHAT_URL, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    err = await resp.text()
                    logger.error(f"Ollama chat error {resp.status}: {err}")
                    return {"message": {"role": "assistant", "content": "I apologize, but I am currently having trouble processing that query."}}
        except Exception as e:
            logger.error(f"Ollama chat connection error: {e}")
            return {"message": {"role": "assistant", "content": "I apologize, but I am currently unable to reach the language model."}}

    def normalize_response(self, transcript: str, res_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes whatever Ollama returns (plain content or tool_calls) into the exact contract:
        {
          "transcript": "",
          "type": "speak" | "tool_call",
          "text": "<spoken text -- also what you will synthesize into the following audio>",
          "tool_call": {"name": "...", "args": {...}} // only present if type is tool_call
        }
        """
        message = res_json.get("message", {})
        raw_content = message.get("content", "").strip()
        tool_calls = message.get("tool_calls", [])

        # Clean thinking tags, markdown, asterisks, and emojis
        clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL)
        clean_content = re.sub(r"[\*\#\_\[\]\`]", "", clean_content)
        clean_content = re.sub(r"[\U00010000-\U0010ffff]", "", clean_content).strip()

        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if name in self.allowed_tool_names:
                spoken = clean_content
                if not spoken:
                    # Provide natural spoken feedback if Ollama did not provide content
                    if name == "navigate_to":
                        loc = args.get("location", "the destination")
                        spoken = f"Navigating to {loc} now."
                    elif name == "navigate_and_deliver":
                        loc = args.get("location", "the destination")
                        item = args.get("item", "the item")
                        spoken = f"Delivering {item} to {loc}."
                    elif name == "move_distance":
                        meters = float(args.get("meters", 1.0))
                        dist_str = f"{meters:.1f}".rstrip('0').rstrip('.')
                        if meters >= 0:
                            spoken = f"Moving forward {dist_str} meters."
                        else:
                            spoken = f"Moving backward {abs(meters):.1f} meters."
                    elif name == "rotate":
                        degs = float(args.get("degrees", 0.0))
                        deg_str = f"{degs:.1f}".rstrip('0').rstrip('.')
                        if degs >= 0:
                            spoken = f"Rotating {deg_str} degrees counterclockwise."
                        else:
                            spoken = f"Rotating {abs(degs):.1f} degrees clockwise."
                    elif name == "stop":
                        spoken = "Stopping all robot motion immediately."
                    else:
                        spoken = f"Executing {name}."

                return {
                    "transcript": transcript,
                    "type": "tool_call",
                    "text": spoken,
                    "tool_call": {
                        "name": name,
                        "args": args
                    }
                }

        # Plain speak response
        spoken = clean_content if clean_content else "I am here. How can I assist you?"
        return {
            "transcript": transcript,
            "type": "speak",
            "text": spoken
        }

    async def process_utterance(self, ws: web.WebSocketResponse, audio_data: bytes, session_id: str = ""):
        """Processes an audio utterance through STT -> RAG -> Ollama Chat (tools) -> Piper TTS"""
        total_t0 = time.time()
        logger.info(f"Incoming audio utterance: {len(audio_data)} bytes (session: '{session_id}')")

        # 1. STT via Whisper CUDA
        transcript, stt_ms = await self.stt.transcribe(audio_data)

        # Anti-hallucination check
        cleaned_transcript = transcript.lower().strip()
        if not cleaned_transcript or cleaned_transcript in HALLUCINATIONS or len(cleaned_transcript) <= 2:
            logger.info(f"Ignoring silent/hallucinated transcript: '{transcript}'")
            return

        logger.info(f"✓ STT Result ({stt_ms:.1f}ms): '{transcript}'")

        # 2. Knowledge RAG
        rag_t0 = time.time()
        context = self.rag.format_context(transcript)
        rag_ms = (time.time() - rag_t0) * 1000.0
        logger.info(f"✓ RAG Context retrieved in {rag_ms:.1f}ms (matches: {bool(context.strip())})")

        # 3. Ollama Chat with Tools
        llm_t0 = time.time()
        chat_res = await self.chat_with_tools(transcript, context)
        llm_ms = (time.time() - llm_t0) * 1000.0

        # 4. Normalize to exact JSON contract
        contract_data = self.normalize_response(transcript, chat_res)
        logger.info(f"✓ LLM ({llm_ms:.1f}ms) -> [{contract_data['type']}] text: '{contract_data['text']}'" +
                    (f" | tool: {contract_data.get('tool_call')}" if contract_data['type'] == 'tool_call' else ""))

        # 5. Synthesize speech using Piper (WAV bytes)
        tts_t0 = time.time()
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, self.tts.synthesize_sentence_wav, contract_data["text"])
        tts_ms = (time.time() - tts_t0) * 1000.0
        total_ms = (time.time() - total_t0) * 1000.0

        logger.info(f"✓ TTS ({tts_ms:.1f}ms, {len(wav_bytes)} audio bytes). Turn complete in {total_ms:.1f}ms")

        # 6. Send back: first the JSON message, then the synthesized audio as binary, in that order
        if not ws.closed:
            await ws.send_str(json.dumps(contract_data))
            await ws.send_bytes(wav_bytes)

    async def process_text_query(self, ws: web.WebSocketResponse, query: str):
        """Processes a direct text query through RAG -> Ollama Chat (tools) -> Piper TTS"""
        total_t0 = time.time()
        logger.info(f"Incoming text query: '{query}'")

        context = self.rag.format_context(query)
        chat_res = await self.chat_with_tools(query, context)
        contract_data = self.normalize_response(query, chat_res)

        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, self.tts.synthesize_sentence_wav, contract_data["text"])
        total_ms = (time.time() - total_t0) * 1000.0

        logger.info(f"✓ Text query complete in {total_ms:.1f}ms -> [{contract_data['type']}] text: '{contract_data['text']}'")

        if not ws.closed:
            await ws.send_str(json.dumps(contract_data))
            await ws.send_bytes(wav_bytes)

    # ================= HTTP & WebSocket Route Handlers =================

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Handles incoming WebSocket connections from Orin Nano or Web clients"""
        ws = web.WebSocketResponse(
            max_msg_size=16 * 1024 * 1024,
            compress=False,
            heartbeat=20.0
        )
        await ws.prepare(request)
        client_ip = request.remote
        logger.info(f"Client connected from {client_ip}")

        # Send welcome message
        await ws.send_json({
            "type": "session_ready",
            "version": "2.0",
            "robot": "D-Bot",
            "piper_voice": "en_US-hfc_female-medium",
            "whisper_model": "large-v3-turbo (CUDA)",
            "llm_model": OLLAMA_MODEL,
            "tools_available": list(self.allowed_tool_names)
        })

        audio_buffer = bytearray()
        is_streaming_audio = False

        try:
            async for msg in ws:
                if logger.isEnabledFor(logging.DEBUG):
                    data_len = len(msg.data) if hasattr(msg, "data") and msg.data is not None else 0
                    logger.debug(f"WS [{client_ip}] {msg.type.name} frame: {data_len} bytes")

                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    session_id = ""
                    audio_payload = data

                    # Unpack format: [4 bytes little-endian uint32 header length] + [header JSON bytes] + [raw PCM audio bytes]
                    if len(data) >= 4:
                        hdr_len = struct.unpack("<I", data[:4])[0]
                        if 0 < hdr_len < len(data) - 4:
                            try:
                                hdr_json = json.loads(data[4:4 + hdr_len].decode("utf-8"))
                                session_id = hdr_json.get("session_id", "")
                                audio_payload = data[4 + hdr_len:]
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug(f"Unpacked header JSON ({hdr_len} bytes, session_id='{session_id}'), audio: {len(audio_payload)} bytes")
                            except Exception:
                                audio_payload = data

                    if is_streaming_audio:
                        audio_buffer.extend(audio_payload)
                    else:
                        asyncio.create_task(self.process_utterance(ws, audio_payload, session_id))

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
                            query = payload.get("text", "")
                            if query:
                                asyncio.create_task(self.process_text_query(ws, query))

                        elif msg_type == "ping":
                            await ws.send_json({"type": "pong", "time": time.time()})

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {msg.data[:100]}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    exc = ws.exception()
                    logger.error(f"WS error from {client_ip}: {exc}")

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
    parser = argparse.ArgumentParser(description="D-Bot Voice Server on Jetson AGX Thor")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--host", default=HOST, help=f"Host address to bind (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to bind (default: {PORT})")
    args = parser.parse_args()

    if args.debug or os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging mode activated.")

    logger.info("=" * 70)
    logger.info("🤖 Starting D-Bot Voice Server on Jetson AGX Thor...")
    logger.info(f"   Listening on: http://{args.host}:{args.port}")
    logger.info(f"   WebSocket URL: ws://{args.host}:{args.port}/ws")
    logger.info(f"   Debug mode: {'ENABLED' if logger.isEnabledFor(logging.DEBUG) else 'DISABLED'}")
    logger.info(f"   Ollama Model: {OLLAMA_MODEL}")
    logger.info("=" * 70)
    web.run_app(init_app(), host=args.host, port=args.port)
