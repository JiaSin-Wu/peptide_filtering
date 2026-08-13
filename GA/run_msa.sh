#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/msa"
OUTPUT_DIR="$SCRIPT_DIR/msa_outputs"

mkdir -p "$OUTPUT_DIR/logs"

docker run --rm \
  --gpus all \
  -v "$INPUT_DIR":/root/inputs \
  -v "$OUTPUT_DIR":/root/outputs \
  -v /opt/model_parameter:/root/af3_params \
  -v /opt/af3_databases:/root/public_databases \
  -e PYTHONPATH=/app/alphafold/src \
  jiasin/alphafold3:latest python3 /app/alphafold/run_alphafold.py \
  --input_dir=/root/inputs \
  --output_dir=/root/outputs \
  --model_dir=/root/af3_params \
  --run_data_pipeline=true \
  --run_inference=false \
  --force_output_dir=true \
  2>&1 | tee "$OUTPUT_DIR/logs/run_msa.log"

echo "All done. Results in $OUTPUT_DIR"
