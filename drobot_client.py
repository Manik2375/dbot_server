#!/usr/bin/env python3
"""
D-Bot Client for Jetson Orin Nano (Campus Guide Robot)
======================================================
Connects to Jetson AGX Thor Voice Server over WebSockets:
- Captures audio from microphone (or test WAV file)
- Streams audio with low latency to Thor
- Receives sentence-by-sentence synthesized Piper audio chunks
- Plays audio through robot speakers with minimal buffer delay

Usage:
  python drobot_client.py --server ws://<THOR_IP>:8765/ws
  python drobot_client.py --test-file /path/to/test.wav
  python drobot_client.py --text "Where is the Drobotics lab?"
"""

import argparse
import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import time
import wave

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drobot_client")


def play_wav_bytes(wav_data: bytes):
    """Plays WAV audio bytes directly to the system speaker at correct sample rate"""
    import tempfile
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_data)
            temp_path = tf.name

        # pw-play with a file path parses the WAV header (22050 Hz) and plays at normal speed
        res = subprocess.run(["pw-play", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            subprocess.run(["aplay", "-q", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.warning(f"Playback error: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def run_text_query(server_url: str, text: str):
    """Sends a text test query to Thor and plays back the voice response"""
    logger.info(f"Connecting to Thor server at {server_url}...")
    async with websockets.connect(server_url) as ws:
        # Wait for session ready
        init_msg = await ws.recv()
        logger.info(f"Server ready: {init_msg}")

        logger.info(f"Sending query: '{text}'")
        await ws.send(json.dumps({"type": "text_query", "text": text}))

        sentence_idx = 0
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                logger.info(f"🔊 Received sentence {sentence_idx} audio ({len(msg)} bytes). Playing...")
                play_wav_bytes(msg)
                sentence_idx += 1
            else:
                data = json.loads(msg)
                mtype = data.get("type")
                if mtype == "llm_chunk":
                    print(data.get("token", ""), end="", flush=True)
                elif mtype == "turn_complete":
                    print("\n")
                    logger.info(f"✓ Turn complete! Total latency: {data.get('total_latency_ms')}ms")
                    break


async def run_audio_file(server_url: str, file_path: str):
    """Sends an audio WAV file to Thor as a spoken query"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    with open(file_path, "rb") as f:
        audio_data = f.read()

    logger.info(f"Connecting to Thor server at {server_url}...")
    async with websockets.connect(server_url) as ws:
        init_msg = await ws.recv()
        logger.info(f"Server ready: {init_msg}")

        t0 = time.time()
        logger.info(f"Transmitting {len(audio_data)} bytes of audio...")
        await ws.send(audio_data)

        sentence_idx = 0
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                ttfa = (time.time() - t0) * 1000.0
                logger.info(f"🔊 Received audio chunk {sentence_idx} ({len(msg)} bytes, TTFA: {ttfa:.1f}ms). Playing...")
                play_wav_bytes(msg)
                sentence_idx += 1
            else:
                data = json.loads(msg)
                mtype = data.get("type")
                if mtype == "stt_result":
                    logger.info(f"Transcription: '{data.get('transcript')}' (STT took {data.get('latency_ms')}ms)")
                elif mtype == "llm_chunk":
                    print(data.get("token", ""), end="", flush=True)
                elif mtype == "turn_complete":
                    print("\n")
                    logger.info(f"✓ Turn complete! Total latency: {data.get('total_latency_ms')}ms")
                    break


async def run_interactive_mic(server_url: str):
    """Interactive microphone streaming client using PyAudio or arecord"""
    logger.info("Interactive microphone mode")
    logger.info("Connecting to Thor server...")
    async with websockets.connect(server_url) as ws:
        init_msg = await ws.recv()
        logger.info(f"Server ready: {init_msg}")

        print("\n=== D-Bot Voice Client Ready ===")
        print("Press ENTER to record a 4-second voice query, or type 'q' to quit.")

        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "\n[Press ENTER to Speak] > ")
            if user_input.strip().lower() == "q":
                break

            print("🔴 Recording 3.5s from microphone...")
            # Use arecord (native on Jetson)
            proc = await asyncio.create_subprocess_exec(
                "arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "4", "-t", "wav", "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await proc.communicate()

            if not stdout:
                print("⚠ No audio captured.")
                continue

            print("⚡ Sending audio to Thor...")
            t0 = time.time()
            await ws.send(stdout)

            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    play_wav_bytes(msg)
                else:
                    data = json.loads(msg)
                    mtype = data.get("type")
                    if mtype == "stt_result":
                        print(f"👂 You said: \"{data.get('transcript')}\"")
                    elif mtype == "llm_chunk":
                        print(data.get("token", ""), end="", flush=True)
                    elif mtype == "turn_complete":
                        print(f"\n[Total Response Time: {data.get('total_latency_ms')}ms]")
                        break


def main():
    parser = argparse.ArgumentParser(description="D-Bot Orin Nano Voice Client")
    parser.add_argument("--server", default="ws://127.0.0.1:8765/ws", help="WebSocket URL of Jetson Thor voice server")
    parser.add_argument("--test-file", help="Path to WAV file to send as query")
    parser.add_argument("--text", help="Send a test text query")
    parser.add_argument("--interactive", action="store_true", help="Record audio from microphone on demand")
    args = parser.parse_args()

    if args.text:
        asyncio.run(run_text_query(args.server, args.text))
    elif args.test_file:
        asyncio.run(run_audio_file(args.server, args.test_file))
    elif args.interactive:
        asyncio.run(run_interactive_mic(args.server))
    else:
        # Default: demonstrate a query
        print("No mode specified. Running test query...")
        asyncio.run(run_text_query(args.server, "Where is the Drobotics lab located?"))


if __name__ == "__main__":
    main()
