#!/usr/bin/env bash
# ============================================================================
# Whisper.cpp CUDA Server for Jetson AGX Thor (D-Bot Campus Guide)
# ============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/whisper_build/bin/whisper-server"
MODEL="$DIR/models/whisper/ggml-large-v3-turbo-q5_0.bin"
PORT=8080
HOST="0.0.0.0"
THREADS=8

PROMPT="D-Bot, JUIT, Jaypee University of Information Technology, Waknaghat, Solan, Drobotics Lab, ECL, CSE, ECE, Biotechnology, Civil Engineering"

if [ ! -f "$BIN" ]; then
    echo "Error: whisper-server binary not found at $BIN"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL"
    exit 1
fi

echo "Starting Whisper CUDA Server on $HOST:$PORT with model $MODEL..."
exec "$BIN" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -t "$THREADS" \
    --flash-attn \
    --prompt "$PROMPT" \
    --inference-path "/inference" \
    --no-timestamps
