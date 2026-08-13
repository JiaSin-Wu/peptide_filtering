#!/bin/bash
# Start the persistent AF3 daemon container.
# ModelRunner is loaded once; the container stays alive waiting for jobs.
# Usage: bash start_af3_daemon.sh [container-name]

set -e

CONTAINER="${1:-af3_daemon}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AF3_DAEMON_PY="$SCRIPT_DIR/af3_daemon.py"
mkdir -p "$SCRIPT_DIR/GA/input" "$SCRIPT_DIR/GA/outputs"

# Stop existing container if present
if docker inspect "$CONTAINER" &>/dev/null; then
    echo "[daemon] Stopping existing container: $CONTAINER"
    docker rm -f "$CONTAINER"
fi

echo "[daemon] Starting $CONTAINER ..."
docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    -v "$SCRIPT_DIR/GA/input":/root/inputs \
    -v "$SCRIPT_DIR/GA/outputs":/root/outputs \
    -v /opt/model_parameter:/root/af3_params \
    -v "$AF3_DAEMON_PY":/app/alphafold/af3_daemon.py \
    -e PYTHONPATH=/app/alphafold/src \
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
