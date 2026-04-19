#!/usr/bin/env bash
set -e

echo "=== S2VS Video Deduplication ==="

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run ./setup.sh first."
    exit 1
fi
source .venv/bin/activate

# Ensure Qdrant is running
if ! curl -s http://localhost:6333/healthz &>/dev/null; then
    echo "WARNING: Qdrant not reachable at localhost:6333"
    echo "Start with: docker run -d -p 6333:6333 -v \$(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant"
    echo ""
fi

# Suppress duplicate library warnings on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Start backend
echo "Starting backend on http://localhost:8000 ..."
uvicorn api_server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/status &>/dev/null; then
        break
    fi
    sleep 1
done

# Start frontend
echo "Starting frontend on http://localhost:3000 ..."
if command -v bun &>/dev/null; then
    cd web-ui && bun run dev &
else
    cd web-ui && npm run dev &
fi
FRONTEND_PID=$!

echo ""
echo "=== Running ==="
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  Press Ctrl+C to stop"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# Wait for either process to exit
wait
