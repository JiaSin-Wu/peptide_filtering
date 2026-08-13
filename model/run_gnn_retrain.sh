#!/bin/bash
# Run all GNN/EGNN retrain jobs sequentially
# Usage: conda run -n af3_ml bash run_gnn_retrain.sh

set -e
LOG=/home/jiasin/af3_ml/gnn_retrain.log
exec > >(tee -a $LOG) 2>&1

echo "=== GNN RETRAIN START $(date) ==="

echo "[1/9] GAT l2 h4 drop0.1 (default, save_model)"
python3 -m gnn.train --model_type gat --dropout 0.1 --train_all --tag drop01 --save_model

echo "[2/9] GAT l2 h8 drop0.1"
python3 -m gnn.train --model_type gat --n_heads 8 --dropout 0.1 --train_all --tag gat_h8_drop01

echo "[3/9] GAT l2 h16 drop0.1"
python3 -m gnn.train --model_type gat --n_heads 16 --dropout 0.1 --train_all --tag gat_h16_drop01

echo "[4/9] GAT l1 h4 drop0.1"
python3 -m gnn.train --model_type gat --n_layers 1 --dropout 0.1 --train_all --tag gat_l1_drop01

echo "[5/9] Xattn l2 h4 drop0.1"
python3 -m gnn.train --model_type xattn --dropout 0.1 --train_all --tag drop01

echo "[6/9] Xattn l1 h4 drop0.1"
python3 -m gnn.train --model_type xattn --n_layers 1 --dropout 0.1 --train_all --tag xattn_l1_drop01

echo "[7/9] Xattn l1 h8 drop0.1"
python3 -m gnn.train --model_type xattn --n_layers 1 --n_heads 8 --dropout 0.1 --train_all --tag xattn_l1_h8_drop01

echo "[8/9] EGNN l1"
python3 -m egnn.train --n_layers 1 --train_all --tag l1

echo "[9/9] EGNN l2"
python3 -m egnn.train --n_layers 2 --train_all --tag l2

echo "=== GNN RETRAIN DONE $(date) ==="
