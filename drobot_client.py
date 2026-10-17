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
import struct
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


def pack_audio_message(audio_data: bytes, session_id: str = "orin-nano-01") -> bytes:
    """
    Packs audio into binary frame:
    [4 bytes little-endian uint32 header length] + [header JSON bytes] + [audio bytes]
    """
    header_json = json.dumps({"session_id": session_id}).encode("utf-8")
    header_len = struct.pack("<I", len(header_json))
    return header_len + header_json + audio_data


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
    """Sends a text test query to Thor and receives JSON contract + synthesized audio"""
    logger.info(f"Connecting to Thor server at {server_url}...")
    async with websockets.connect(server_url, compression=None) as ws:
        init_msg = await ws.recv()
        logger.info(f"Server ready: {init_msg}")

        logger.info(f"Sending text query: '{text}'")
        t0 = time.time()
        await ws.send(json.dumps({"type": "text_query", "text": text}))

        # Expect first JSON contract, then binary audio
        msg1 = await ws.recv()
        if isinstance(msg1, str):
            res_json = json.loads(msg1)
            logger.info("--------------------------------------------------")
            logger.info(f"Received JSON Contract: {json.dumps(res_json, indent=2)}")
            if res_json.get("type") == "tool_call":
                tc = res_json.get("tool_call", {})
                logger.info(f"🤖 [ROBOT ACTION] Tool Call: {tc.get('name')}({tc.get('args')})")
            logger.info(f"🗣️ [Robot Speech] \"{res_json.get('text')}\"")
            logger.info("--------------------------------------------------")

        msg2 = await ws.recv()
        if isinstance(msg2, bytes):
            total_ms = (time.time() - t0) * 1000.0
            logger.info(f"🔊 Received {len(msg2)} bytes of synthesized speech ({total_ms:.1f}ms). Playing...")
            play_wav_bytes(msg2)


async def run_audio_file(server_url: str, file_path: str):
    """Sends an audio WAV file to Thor as a spoken query using packed binary format"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    with open(file_path, "rb") as f:
        audio_data = f.read()

    # Pack into: [4 bytes header len] + [header JSON] + [audio bytes]
    packed_message = pack_audio_message(audio_data, session_id="test-file-session")

    logger.info(f"Connecting to Thor server at {server_url}...")
    async with websockets.connect(server_url, compression=None) as ws:
        init_msg = await ws.recv()
        logger.info(f"Server ready: {init_msg}")

        t0 = time.time()
        logger.info(f"Transmitting packed audio message ({len(packed_message)} bytes)...")
        await ws.send(packed_message)

        # 1. Expect JSON message
        msg1 = await ws.recv()
        if isinstance(msg1, str):
            res_json = json.loads(msg1)
            logger.info("--------------------------------------------------")
            logger.info(f"Received JSON Contract: {json.dumps(res_json, indent=2)}")
            logger.info(f"👂 [User Spoke] \"{res_json.get('transcript')}\"")
            if res_json.get("type") == "tool_call":
                tc = res_json.get("tool_call", {})
                logger.info(f"🤖 [ROBOT ACTION] Tool Call: {tc.get('name')}({tc.get('args')})")
            logger.info(f"🗣️ [Robot Speech] \"{res_json.get('text')}\"")
            logger.info("--------------------------------------------------")

        # 2. Expect synthesized WAV bytes
        msg2 = await ws.recv()
        if isinstance(msg2, bytes):
            total_ms = (time.time() - t0) * 1000.0
            logger.info(f"🔊 Received {len(msg2)} bytes of synthesized audio ({total_ms:.1f}ms). Playing...")
            play_wav_bytes(msg2)


async def run_interactive_mic(server_url: str):
    """Interactive microphone client using arecord on Jetson"""
    logger.info("Interactive microphone mode")
    logger.info("Connecting to Thor server...")
    async with websockets.connect(server_url, compression=None) as ws:
        init_msg = await ws.recv()
        logger.info(f"Server ready: {init_msg}")

        print("\n=== D-Bot Voice Client Ready ===")
        print("Press ENTER to record a 3.5-second voice query, or type 'q' to quit.")

        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "\n[Press ENTER to Speak] > ")
            if user_input.strip().lower() == "q":
                break

            print("🔴 Recording 3.5s from microphone...")
            proc = await asyncio.create_subprocess_exec(
                "arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "4", "-t", "raw", "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await proc.communicate()

            if not stdout:
                print("⚠ No audio captured.")
                continue

            print("⚡ Sending audio to Thor...")
            packed_data = pack_audio_message(stdout, session_id="live-mic-session")
            t0 = time.time()
            await ws.send(packed_data)

            # Receive JSON contract
            msg1 = await ws.recv()
            if isinstance(msg1, str):
                res_json = json.loads(msg1)
                print("--------------------------------------------------")
                print(f"👂 You said: \"{res_json.get('transcript')}\"")
                if res_json.get("type") == "tool_call":
                    tc = res_json.get("tool_call", {})
                    print(f"🤖 Action: {tc.get('name')}({tc.get('args')})")
                print(f"🗣️ D-Bot: \"{res_json.get('text')}\"")
                print("--------------------------------------------------")

            # Receive synthesized audio
            msg2 = await ws.recv()
            if isinstance(msg2, bytes):
                total_ms = (time.time() - t0) * 1000.0
                print(f"[Latency: {total_ms:.1f}ms] Playing robot speech...")
                play_wav_bytes(msg2)


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
