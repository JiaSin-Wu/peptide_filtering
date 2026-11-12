#!/bin/bash
# Start the persistent AF3 daemon container.
# ModelRunner is loaded once; the container stays alive waiting for jobs.
# Usage: bash start_af3_daemon.sh [container-name]
# GPU memory cap (e.g. on a shared machine): MEM_FRACTION=0.45 bash start_af3_daemon.sh
# Model weights elsewhere (e.g. /opt/model_parameter empty/root-owned/no sudo):
#   MODEL_DIR=/data1/jiasin/af3/af3_model_parameter bash start_af3_daemon.sh

set -e

CONTAINER="${1:-af3_daemon}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
MODEL_DIR="${MODEL_DIR:-/opt/model_parameter}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AF3_DAEMON_PY="$SCRIPT_DIR/af3_daemon.py"
mkdir -p "$SCRIPT_DIR/GA/af3/infer_input" "$SCRIPT_DIR/GA/af3/infer_output"

# Stop existing container if present
if docker inspect "$CONTAINER" &>/dev/null; then
    echo "[daemon] Stopping existing container: $CONTAINER"
    docker rm -f "$CONTAINER"
fi

echo "[daemon] Starting $CONTAINER (GPU mem fraction: $MEM_FRACTION) ..."
docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    -v "$SCRIPT_DIR/GA/af3/infer_input":/root/inputs \
    -v "$SCRIPT_DIR/GA/af3/infer_output":/root/outputs \
    -v "$MODEL_DIR":/root/af3_params \
    -v "$AF3_DAEMON_PY":/app/alphafold/af3_daemon.py \
    -e PYTHONPATH=/app/alphafold/src \
    -e XLA_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
    jiasin/alphafold3:latest \
    python3 /app/alphafold/af3_daemon.py

echo "[daemon] Container started. Waiting for model to load..."
echo "(tail logs with: docker logs -f $CONTAINER)"

# Wait until "Model ready" appears in logs (up to 5 min)
TIMEOUT=300
ELAPSED=0
while ! docker logs "$CONTAINER" 2>&1 | grep -q "Model ready"; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[daemon] ERROR: model did not load within ${TIMEOUT}s"
        echo "[daemon] Check: docker logs $CONTAINER"
        exit 1
    fi
done

echo "[daemon] AF3 daemon ready (container: $CONTAINER)"
