#!/bin/bash
# Wait for graph rebuild to finish, then train both GATv2 and CrossAttn

echo "Waiting for one-hot graph build..."
while ! grep -q "Done\." /data1/jiasin/af3/ml/precompute_onehot.log 2>/dev/null; do
    sleep 30
done
echo "Graphs ready. Starting training..."

# GATv2 with one-hot nodes
conda run -n af3ml python3 -m ml.gnn.train \
    --model_type gat --both_labels \
    --graph_dir ml/data/graphs_onehot \
    --n_folds 10 --n_models 3 --epochs 50 --batch 16 \
    --tag onehot \
    2>&1 | tee ml/gat_onehot.log &

# CrossAttn with one-hot nodes
conda run -n af3ml python3 -m ml.gnn.train \
    --model_type xattn --both_labels \
    --graph_dir ml/data/graphs_onehot \
    --n_folds 10 --n_models 3 --epochs 50 --batch 16 \
    --tag onehot \
    2>&1 | tee ml/xattn_onehot.log &

wait
echo "Both done."
