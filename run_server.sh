#!/usr/bin/env bash
# ============================================================================
# Master Launcher for D-Bot Voice Pipeline on Jetson AGX Thor
# ============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Parse command-line options
DEBUG_MODE=false
for arg in "$@"; do
    case $arg in
	    --debug|-d)
            DEBUG_MODE=true
            ;;
    esac
done

echo "============================================================================"
if [ "$DEBUG_MODE" = true ]; then
    echo "🚀 Initializing D-Bot Voice System on Jetson AGX Thor [DEBUG MODE]"
else
    echo "🚀 Initializing D-Bot Campus Guide Voice System on Jetson AGX Thor"
fi
echo "============================================================================"

# 1. Check Ollama
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    echo "Starting Ollama service..."
    if [ "$DEBUG_MODE" = true ]; then
        ollama serve 2>&1 | sed -u 's/^/[OLLAMA] /' &
    else
        ollama serve > /dev/null 2>&1 &
    fi
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

# 3. Stream background logs in debug mode and run Voice Server
if [ "$DEBUG_MODE" = true ]; then
    echo "Attaching real-time Whisper log stream ([WHISPER])..."
    tail -n 0 -F "$DIR/whisper_server.log" 2>/dev/null | sed -u 's/^/[WHISPER] /' &
    TAIL_PID=$!

    cleanup() {
        echo -e "\nShutting down D-Bot services..."
        kill "$TAIL_PID" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    echo "Starting Main Voice WebSocket Server on port 8765 (DEBUG)..."
    "./voice_agent/bin/python" "$DIR/voice_server.py" --debug
else
    echo "Starting Main Voice WebSocket Server on port 8765..."
    exec "./voice_agent/bin/python" "$DIR/voice_server.py"
fi

