#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/input"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

mkdir -p "$OUTPUT_DIR/logs"

docker run --rm \
  --gpus all \
  -v "$INPUT_DIR":/root/inputs \
  -v "$OUTPUT_DIR":/root/outputs \
  -v /opt/model_parameter:/root/af3_params \
  -e PYTHONPATH=/app/alphafold/src \
  jiasin/alphafold3:latest python3 /app/alphafold/run_alphafold.py \
  --input_dir=/root/inputs \
  --output_dir=/root/outputs \
  --model_dir=/root/af3_params \
  --run_data_pipeline=false \
  --run_inference=true \
  --num_diffusion_samples=1 \
  --save_embeddings=true \
  --force_output_dir=true \
  2>&1 | tee "$OUTPUT_DIR/logs/run_infer.log"

echo "All done. Results in $OUTPUT_DIR"
