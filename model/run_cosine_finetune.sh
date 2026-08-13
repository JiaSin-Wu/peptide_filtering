#!/bin/bash
# Wait for cosine pretrain to finish, then fine-tune

echo "Waiting for cosine pretrain to complete..."
while ! grep -q "Best val loss" /data1/jiasin/af3/ml/pretrain_cosine.log 2>/dev/null; do
    sleep 60
done

echo "Cosine pretrain done. Copying encoder weights..."
cp /data1/jiasin/af3/ml/results/pretrain_encoder.pt \
   /data1/jiasin/af3/ml/results/pretrain_encoder_cosine.pt

echo "Starting fine-tune with cosine pretrained encoder..."
conda run -n af3ml python3 -m ml.gnn.train \
    --model_type xattn --both_labels \
    --pretrained_encoder ml/results/pretrain_encoder_cosine.pt \
    --n_folds 10 --n_models 3 --epochs 50 --batch 16 \
    --tag pretrain_cosine \
    2>&1 | tee /data1/jiasin/af3/ml/xattn_pretrained_cosine.log

echo "All done."
