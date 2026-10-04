#!/usr/bin/env bash
# ============================================================================
# Master Launcher for D-Bot Voice Pipeline on Jetson AGX Thor
# ============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "============================================================================"
echo "🚀 Initializing D-Bot Campus Guide Voice System on Jetson AGX Thor"
echo "============================================================================"

# 1. Check Ollama
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 2
fi
echo "✓ Ollama service is active"

# 2. Check Whisper CUDA Server
if ! curl -s http://127.0.0.1:8080/ > /dev/null && ! nc -z 127.0.0.1 8080 2>/dev/null; then
    echo "Starting Whisper CUDA Server on port 8080..."
    "$DIR/start_whisper_server.sh" > "$DIR/whisper_server.log" 2>&1 &
    sleep 3
fi
echo "✓ Whisper CUDA server is active on port 8080"

# 3. Start Voice Server
echo "Starting Main Voice WebSocket Server on port 8765..."
exec "./voice agent/venv/bin/python" "$DIR/voice_server.py"
